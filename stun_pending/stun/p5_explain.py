"""P5: why does the clean model's two-index router raise held-out accuracy while recall falls?

Run with --retriever pointing at the clean BGE checkpoint. Three analyses:

1. Cross of candidate source x rendering:
     source   : description index (docs) vs two-index router
     rendering: description only vs description + experience lines
   If the gain needs the experience rendering, the reader is using experience text.
   If it needs the two-index candidates, the gain comes from which tools are proposed.
2. Gold in both lists: on held-out queries whose gold tool is in both candidate lists,
   compare accuracy. A gain there means different distractors, not different reach.
3. Position randomization: rerank each list with candidates shuffled R times (accuracy is
   the mean over shuffles). If the gain disappears, it came from candidate order.
"""
from __future__ import annotations

import json
from collections import defaultdict

from .common import concat_experience, default_render, description_entries, load_client_experience, seed_everything, summarize, two_index, walk_down
from .runner import Index, base_args, exp_text_by_tool, load_common, paired_report, run_arm, save


def main():
    p = base_args(__doc__)
    p.add_argument("--shuffles", type=int, default=5)
    a = p.parse_args()
    catalog, tests, H, seeds, rer = load_common(a)
    desc = description_entries(catalog)
    D = Index(desc, a.retriever)
    rows = []
    for seed in seeds:
        seed_everything(seed)
        exp = concat_experience(load_client_experience(f"{a.clients_root}/seed{seed}"))
        E = Index(exp, a.retriever)
        r_doc, r_exp = default_render(catalog), default_render(catalog, exp_text_by_tool(exp))
        src = {"docs": lambda q: walk_down(D.scores(q.text), D.entries),
               "twoidx": lambda q: two_index(D.scores(q.text), D.entries, E.scores(q.text), E.entries)}
        for s_name, f in src.items():
            for rd_name, rd in (("descrender", r_doc), ("exprender", r_exp)):
                rows += run_arm(f"{s_name}|{rd_name}", seed, tests, H, f, rer, rd)
            rows += run_arm(f"{s_name}|exprender|shuffled", seed, tests, H, f, rer, r_exp, shuffle_positions=a.shuffles)
    arms = sorted({r["arm"] for r in rows})
    summ = [summarize(n, [r for r in rows if r["arm"] == n]) for n in arms]
    # gold in both lists
    by = defaultdict(dict)
    for r in rows:
        if r["stratum"] == "heldout" and r["arm"] in ("docs|exprender", "twoidx|exprender"):
            by[(r["seed"], r["qid"])][r["arm"]] = r
    both = [v for v in by.values() if len(v) == 2 and all(v[k]["r5"] for k in v)]
    gib = {"n_gold_in_both": len(both),
           "acc_docs": sum(v["docs|exprender"]["acc"] for v in both) / max(1, len(both)),
           "acc_twoidx": sum(v["twoidx|exprender"]["acc"] for v in both) / max(1, len(both))}
    tests_ = [paired_report(rows, "twoidx|exprender", "docs|exprender"), paired_report(rows, "twoidx|descrender", "docs|descrender"),
              paired_report(rows, "twoidx|exprender|shuffled", "docs|exprender|shuffled"), {"gold_in_both": gib}]
    save(a.out, rows, summ, tests_)


if __name__ == "__main__":
    main()
