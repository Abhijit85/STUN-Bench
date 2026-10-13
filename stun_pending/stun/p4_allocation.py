"""P4: slot-budget sweep for the two-index router, chosen on a tool-disjoint development set.

Per seed, held-out tools are split 50/50 into dev and test tools. Held-out queries go to
dev or test by their gold tools (queries whose gold tools straddle both halves are dropped).
Labeled queries are split 50/50 at random. The budget (description slots out of 5) that
maximizes dev accuracy over held-out + labeled dev queries is chosen; we report every
budget on the test queries, marking the chosen one, so the choice is visible.

Budgets: 5+0 (descriptions only), 4+1, 3+2 (paper default), 2+3, 1+4.
"""
from __future__ import annotations

import json
import random

from .common import concat_experience, default_render, description_entries, load_client_experience, seed_everything, stratum_of, summarize, two_index
from .runner import Index, base_args, exp_text_by_tool, load_common, run_arm, save


def split_queries(tests, H, seed):
    rng = random.Random(seed)
    tools = sorted(H)
    rng.shuffle(tools)
    dev_tools = set(tools[: len(tools) // 2])
    dev, test = [], []
    for q in tests:
        s = stratum_of(q, H)
        if s == "heldout":
            inside = {g in dev_tools for g in q.gold}
            if inside == {True}:
                dev.append(q)
            elif inside == {False}:
                test.append(q)
        else:
            (dev if rng.random() < 0.5 else test).append(q)
    return dev, test


def main():
    p = base_args(__doc__)
    a = p.parse_args()
    catalog, tests, H, seeds, rer = load_common(a)
    desc = description_entries(catalog)
    D = Index(desc, a.retriever)
    rows, chosen = [], {}
    for seed in seeds:
        seed_everything(seed)
        exp = concat_experience(load_client_experience(f"{a.clients_root}/seed{seed}"))
        E = Index(exp, a.retriever)
        ren = default_render(catalog, exp_text_by_tool(exp))
        dev, test = split_queries(tests, H, seed)
        dev_acc = {}
        for nd in (5, 4, 3, 2, 1):
            f = lambda q, nd=nd: two_index(D.scores(q.text), D.entries, E.scores(q.text), E.entries, n_desc=nd)
            r_dev = run_arm(f"{nd}+{5 - nd}", seed, dev, H, f, rer, ren)
            dev_acc[nd] = sum(r["acc"] for r in r_dev) / max(1, len(r_dev))
            for r in run_arm(f"{nd}+{5 - nd}", seed, test, H, f, rer, ren):
                r["split"] = "test"
                rows.append(r)
        best = max(dev_acc, key=lambda k: (dev_acc[k], k))
        chosen[seed] = {"chosen": f"{best}+{5 - best}", "dev_acc": dev_acc, "n_dev": len(dev), "n_test": len(test)}
        print(json.dumps({"seed": seed, **chosen[seed]}))
    summ = [summarize(n, [r for r in rows if r["arm"] == n]) for n in ("5+0", "4+1", "3+2", "2+3", "1+4")]
    save(a.out, rows, summ, [{"chosen_on_dev": chosen}])


if __name__ == "__main__":
    main()
