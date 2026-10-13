"""P2: score normalization inside the shared index, and BM25 with frozen statistics.

Variants on the same concatenated shared index:
  max        : best entry per tool (equals the paper's walk-down ordering; reference)
  mean       : mean entry score per tool (penalizes tools with many weak entries)
  maxcorr    : max minus sigma * E[max of n normals]; removes the expected gain a tool
               gets only from having n entries instead of one
  srcz       : z-score description and experience scores separately per query, then walk down
  bm25frozen : (only with --retriever bm25) IDF and average length taken from the description
               index, so pooled experience cannot change term weights

If crowding survives maxcorr and srcz, it is not a score-scale or entry-count artifact.
"""
from __future__ import annotations

from .common import (BM25, concat_experience, default_render, description_entries, load_client_experience, per_source_z,
                     seed_everything, summarize, tool_scores, top_k_tools, walk_down)
from .runner import Index, base_args, exp_text_by_tool, load_common, paired_report, run_arm, save


def main():
    p = base_args(__doc__)
    a = p.parse_args()
    catalog, tests, H, seeds, rer = load_common(a)
    desc = description_entries(catalog)
    D = Index(desc, a.retriever)
    rows, arms = [], ["docs", "max", "mean", "maxcorr", "srcz"]
    for seed in seeds:
        seed_everything(seed)
        exp = concat_experience(load_client_experience(f"{a.clients_root}/seed{seed}"))
        S = Index(desc + exp, a.retriever)
        ren_d, ren_s = default_render(catalog), default_render(catalog, exp_text_by_tool(exp))
        rows += run_arm("docs", seed, tests, H, lambda q: walk_down(D.scores(q.text), D.entries), rer, ren_d)
        for agg in ("max", "mean", "maxcorr"):
            rows += run_arm(agg, seed, tests, H, lambda q, g=agg: top_k_tools(tool_scores(S.scores(q.text), S.entries, g)), rer, ren_s)
        rows += run_arm("srcz", seed, tests, H, lambda q: walk_down(per_source_z(S.scores(q.text), S.entries), S.entries), rer, ren_s)
        if a.retriever == "bm25":
            F = BM25([e.text for e in desc + exp], frozen_from=[e.text for e in desc])
            rows += run_arm("bm25frozen", seed, tests, H, lambda q: walk_down(F.score(q.text), desc + exp), rer, ren_s)
            if "bm25frozen" not in arms:
                arms.append("bm25frozen")
    summ = [summarize(n, [r for r in rows if r["arm"] == n]) for n in arms]
    tests_ = [paired_report(rows, n, "docs") for n in arms if n != "docs"]
    save(a.out, rows, summ, tests_)


if __name__ == "__main__":
    main()
