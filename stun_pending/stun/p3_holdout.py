"""P3: alternative held-out tool sets, built with the same label and mention rules.

  g2      : H plus every gold tool of the G2-category test queries
  matched : a set the size of H drawn from G1-instruction gold tools (tools ToolBench calls
            seen), matched to H on category and on pool frequency (labeled items per tool)

Outputs, per variant: heldout.json, pool.jsonl (items kept), removal counts, and the
query strata. Then build client experience from the new pool with your existing client
sampler and rerun p0_concat with --heldout pointing at the new file.

python -m stun.p3_holdout --catalog data/catalog.jsonl --tests data/tests.jsonl \
  --heldout data/heldout.json --pool data/leak_filtered_pool.jsonl --variant g2 --out data/holdout_g2
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from .common import load_catalog, load_tests, read_jsonl, strata, write_jsonl


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def apply_rules(pool: list[dict], H: set[str], catalog: dict) -> tuple[list[dict], dict]:
    """Label rule: any label in H. Mention rule: a normalized held-out tool name appears in the
    query or in the item's tool, API or category metadata (field 'meta_text')."""
    names = {norm(catalog[t].get("name", t)) for t in H if norm(catalog[t].get("name", t))}
    pat = re.compile(r"\b(" + "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True)) + r")\b") if names else None
    kept, n_label, n_mention = [], 0, 0
    for it in pool:
        if any(l in H for l in it["labels"]):
            n_label += 1
            continue
        hay = norm(it["query"] + " " + it.get("meta_text", ""))
        if pat and pat.search(hay):
            n_mention += 1
            continue
        kept.append(it)
    return kept, {"label_rule": n_label, "mention_rule": n_mention, "kept": len(kept)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", required=True)
    p.add_argument("--tests", required=True)
    p.add_argument("--heldout", required=True)
    p.add_argument("--pool", required=True, help="leak-filtered pool: item_id, query, labels[], meta_text")
    p.add_argument("--variant", choices=["g2", "matched"], required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    cat, tests = load_catalog(a.catalog), load_tests(a.tests)
    H = set(json.load(open(a.heldout)))
    pool = read_jsonl(a.pool)
    freq = Counter(l for it in pool for l in it["labels"])
    if a.variant == "g2":
        H2 = H | {g for q in tests if q.group == "G2_category" for g in q.gold}
    else:
        rng = random.Random(a.seed)
        cands = sorted({g for q in tests if q.group == "G1_instruction" for g in q.gold} - H)
        by_cat = defaultdict(list)
        for t in cands:
            by_cat[cat[t].get("category", "")].append(t)
        H2, used = set(), set()
        for t in sorted(H):
            pool_c = [c for c in by_cat.get(cat[t].get("category", ""), []) if c not in used] or [c for c in cands if c not in used]
            if not pool_c:
                break
            best = min(pool_c, key=lambda c: (abs(freq[c] - freq[t]), rng.random()))
            H2.add(best)
            used.add(best)
    kept, counts = apply_rules(pool, H2, cat)
    st = strata(tests, H2)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    json.dump(sorted(H2), open(f"{a.out}/heldout.json", "w"))
    write_jsonl(f"{a.out}/pool.jsonl", kept)
    info = {"variant": a.variant, "H_size": len(H2), **counts, **{f"queries_{k}": len(v) for k, v in st.items()}}
    if a.variant == "matched":
        info["freq_H_median"] = sorted(freq[t] for t in H)[len(H) // 2]
        info["freq_H2_median"] = sorted(freq[t] for t in H2)[len(H2) // 2]
    json.dump(info, open(f"{a.out}/info.json", "w"), indent=2)
    print(json.dumps(info, indent=2))
    assert not {l for it in kept for l in it["labels"]} & H2, "label rule failed"


if __name__ == "__main__":
    main()
