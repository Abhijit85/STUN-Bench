"""P0: rerun the split's headline arms with a concatenated pool.

Arms: description index, shared index (descriptions + every client item as its
own entry), two-index router (3 description + 2 experience slots).
Run once per retriever you report in the paper (Jina-v2, BGE-base, BM25,
Qwen3-Embedding-8B, the clean BGE checkpoint, ToolRet-e5, ToolBench-IR).

python -m stun.p0_concat --catalog data/catalog.jsonl --tests data/tests.jsonl \
  --heldout data/heldout.json --clients-root data/clients --out results/p0_jina
"""
from __future__ import annotations

from .common import concat_experience, default_render, description_entries, load_client_experience, seed_everything, summarize, two_index, walk_down
from .runner import Index, base_args, exp_text_by_tool, load_common, paired_report, run_arm, save


def main():
    p = base_args(__doc__)
    p.add_argument("--n-desc", type=int, default=3, help="description slots of the two-index router")
    a = p.parse_args()
    catalog, tests, H, seeds, rer = load_common(a)
    desc = description_entries(catalog)
    D = Index(desc, a.retriever)
    rows = []
    for seed in seeds:
        seed_everything(seed)
        clients = load_client_experience(f"{a.clients_root}/seed{seed}")
        exp = concat_experience(clients)
        leak = {e.tool_id for e in exp} & H
        assert not leak, f"held-out tools appear in client experience: {sorted(leak)[:5]}"
        E = Index(exp, a.retriever)
        S = Index(desc + exp, a.retriever)
        render_docs = default_render(catalog)
        render_exp = default_render(catalog, exp_text_by_tool(exp))
        rows += run_arm("docs", seed, tests, H, lambda q: walk_down(D.scores(q.text), D.entries), rer, render_docs)
        rows += run_arm("shared_concat", seed, tests, H, lambda q: walk_down(S.scores(q.text), S.entries), rer, render_exp)
        rows += run_arm("two_index", seed, tests, H,
                        lambda q: two_index(D.scores(q.text), D.entries, E.scores(q.text), E.entries, n_desc=a.n_desc), rer, render_exp)
    summ = [summarize(n, [r for r in rows if r["arm"] == n]) for n in ("docs", "shared_concat", "two_index")]
    tests_ = [paired_report(rows, "shared_concat", "docs", s) for s in ("heldout", "labeled", "labeled_clean")]
    tests_ += [paired_report(rows, "two_index", "docs", s) for s in ("heldout", "labeled", "labeled_clean")]
    save(a.out, rows, summ, tests_)


if __name__ == "__main__":
    main()
