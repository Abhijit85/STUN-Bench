"""P1 step 2: matched-content control.

Every index holds all descriptions. Extra entries are added under a 2x2 design:
  style    : query vs desc (same content, from p1_generate pairs)
  coverage : used    = only tools with client experience get entries (as in the real shared index)
             all     = every tool gets entries (held-out tools get generated ones)
Each covered tool gets exactly --n entries, each truncated to --words words, and
pairs shorter than that in either style are dropped, so count and length are
fixed across cells. Also reported: the real shared index (concat) for reference.

Reading the result
  style effect     = used/query minus used/desc  (held-out accuracy and recall)
  coverage effect  = used/query minus all/query
If crowding is only an imbalance in which tools get entries, the 'all' cells
should be close to the description index. If query style matters, used/query
should crowd more than used/desc.
"""
from __future__ import annotations

import json
from collections import defaultdict

from .common import Entry, concat_experience, default_render, description_entries, load_client_experience, read_jsonl, seed_everything, summarize, walk_down
from .runner import Index, base_args, exp_text_by_tool, load_common, paired_report, run_arm, save


def build_cell(gen: list[dict], style: str, coverage: str, used: set[str], n: int, words: int) -> list[Entry]:
    pairs = defaultdict(dict)
    for r in gen:
        pairs[r["pair_id"]][r["style"]] = r
    per_tool = defaultdict(list)
    for pid in sorted(pairs):
        pr = pairs[pid]
        if "query" not in pr or "desc" not in pr:
            continue
        if min(len(pr["query"]["text"].split()), len(pr["desc"]["text"].split())) < words:
            continue  # keep length fixed across styles
        t = pr[style]["tool_id"]
        if coverage == "used" and t not in used:
            continue
        if len(per_tool[t]) < n:
            per_tool[t].append(" ".join(pr[style]["text"].split()[:words]))
    out = []
    for t, texts in per_tool.items():
        if len(texts) == n:  # equal count per covered tool
            out += [Entry(t, x, "exp", {"style": style}) for x in texts]
    return out


def main():
    p = base_args(__doc__)
    p.add_argument("--gen-root", required=True, help="dir with seed42/gen.jsonl ... from p1_generate")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--words", type=int, default=12, help="set to roughly the median word count of real experience items")
    a = p.parse_args()
    catalog, tests, H, seeds, rer = load_common(a)
    desc = description_entries(catalog)
    rows, arms = [], ["docs", "real_concat"]
    D = Index(desc, a.retriever)
    for seed in seeds:
        seed_everything(seed)
        exp = concat_experience(load_client_experience(f"{a.clients_root}/seed{seed}"))
        used = {e.tool_id for e in exp}
        gen = read_jsonl(f"{a.gen_root}/seed{seed}/gen.jsonl")
        rows += run_arm("docs", seed, tests, H, lambda q: walk_down(D.scores(q.text), D.entries), rer, default_render(catalog))
        S = Index(desc + exp, a.retriever)
        rows += run_arm("real_concat", seed, tests, H, lambda q: walk_down(S.scores(q.text), S.entries), rer, default_render(catalog, exp_text_by_tool(exp)))
        for cov in ("used", "all"):
            for sty in ("query", "desc"):
                extra = build_cell(gen, sty, cov, used, a.n, a.words)
                name = f"{cov}_{sty}"
                if name not in arms:
                    arms.append(name)
                print(json.dumps({"seed": seed, "cell": name, "entries": len(extra), "tools": len({e.tool_id for e in extra}),
                                  "heldout_tools_covered": len({e.tool_id for e in extra} & H)}))
                X = Index(desc + extra, a.retriever)
                rows += run_arm(name, seed, tests, H, lambda q: walk_down(X.scores(q.text), X.entries), rer,
                                default_render(catalog, exp_text_by_tool(extra)))
    summ = [summarize(n, [r for r in rows if r["arm"] == n]) for n in arms]
    tests_ = [paired_report(rows, "used_query", "used_desc"), paired_report(rows, "used_query", "all_query"),
              paired_report(rows, "all_query", "docs"), paired_report(rows, "all_desc", "docs")]
    save(a.out, rows, summ, tests_)


if __name__ == "__main__":
    main()
