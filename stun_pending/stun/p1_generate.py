"""P1 step 1: write the extra entries for the matched-content control.

For every tool we need N entries in two styles with the same content:
  query style        : how a user would ask for the tool ("Get flood alerts near the Elbe")
  description style  : how documentation states the same use ("Returns flood alerts for a river region")
Used tools: rewrite each real experience item into description style (the
query-style version is the real item). Held-out tools: generate both styles from
the tool's description, since no client has experience for them.

The generator is a frozen LLM (default Llama-3.1-8B-Instruct, the paper's
expansion generator). Output: gen.jsonl rows
  {tool_id, style: 'query'|'desc', text, origin: 'real'|'rewrite'|'generated', pair_id}
Items with the same pair_id carry the same content in the two styles.
"""
from __future__ import annotations

import argparse
import json

from .common import concat_experience, load_catalog, load_client_experience, write_jsonl

REWRITE = ("Rewrite the following user request about a tool as one sentence of tool documentation that states the "
           "same use, in the third person, starting with a verb such as 'Returns' or 'Finds'. Keep every concrete "
           "detail. Do not add new information.\n\nTool: {name}\nRequest: {text}\nDocumentation sentence:")
GEN_QUERY = ("Here is a tool's documentation. Write {n} different realistic user requests that this tool answers, "
             "one per line, each one sentence, with concrete details.\n\nTool: {name}\nDocumentation: {desc}\nRequests:")
GEN_DESC = ("Here is a tool's documentation. For each of these user requests, write one documentation sentence that "
            "states the same use, in the third person, starting with a verb. One per line, same order.\n\n"
            "Tool: {name}\nDocumentation: {desc}\nRequests:\n{reqs}\nDocumentation sentences:")


class Gen:
    def __init__(self, model: str, device: str = "cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model)
        self.m = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16, device_map=device)
        self.torch = torch

    def __call__(self, prompt: str, max_new_tokens: int = 256) -> str:
        msgs = [{"role": "user", "content": prompt}]
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(self.m.device)
        with self.torch.no_grad():
            out = self.m.generate(ids, max_new_tokens=max_new_tokens, do_sample=False)
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", required=True)
    p.add_argument("--heldout", required=True)
    p.add_argument("--clients-dir", required=True, help="one seed's client dir; run once per seed")
    p.add_argument("--n", type=int, default=5, help="entries per tool (match the equal-count cap of 5)")
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    cat = load_catalog(a.catalog)
    H = set(json.load(open(a.heldout)))
    exp = concat_experience(load_client_experience(a.clients_dir))
    g = Gen(a.model)
    rows, pid = [], 0
    by_tool = {}
    for e in exp:
        by_tool.setdefault(e.tool_id, []).append(e.text)
    for t, texts in by_tool.items():  # used tools: real query style + rewritten description style
        for txt in texts[: a.n]:
            rows.append({"tool_id": t, "style": "query", "text": txt, "origin": "real", "pair_id": pid})
            rows.append({"tool_id": t, "style": "desc", "text": g(REWRITE.format(name=cat[t].get("name", t), text=txt), 96),
                         "origin": "rewrite", "pair_id": pid})
            pid += 1
    for t in sorted(set(cat) - set(by_tool)):  # unused tools (held-out and others): generate both styles
        reqs = [l.strip("-• ").strip() for l in g(GEN_QUERY.format(n=a.n, name=cat[t].get("name", t), desc=cat[t]["description"])).splitlines() if l.strip()][: a.n]
        descs = [l.strip("-• ").strip() for l in g(GEN_DESC.format(name=cat[t].get("name", t), desc=cat[t]["description"],
                                                                   reqs="\n".join(reqs))).splitlines() if l.strip()][: len(reqs)]
        for q, d in zip(reqs, descs):
            rows.append({"tool_id": t, "style": "query", "text": q, "origin": "generated", "pair_id": pid, "heldout": t in H})
            rows.append({"tool_id": t, "style": "desc", "text": d, "origin": "generated", "pair_id": pid, "heldout": t in H})
            pid += 1
    write_jsonl(a.out, rows)
    print(f"wrote {len(rows)} rows for {len(by_tool)} used and {len(set(cat) - set(by_tool))} unused tools")


if __name__ == "__main__":
    main()
