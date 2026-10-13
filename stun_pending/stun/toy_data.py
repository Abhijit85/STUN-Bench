"""Synthetic data to smoke-test every script without a GPU or the real pool.
Numbers from these files mean nothing.

python -m stun.toy_data --out toy
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .common import write_jsonl

WORDS = "weather flood river alert stock price news sports music movie travel hotel flight recipe food map route traffic crypto bank email".split()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="toy")
    a = p.parse_args()
    rng = random.Random(0)
    out = Path(a.out)
    tools = []
    for i in range(40):
        w = rng.sample(WORDS, 3)
        tools.append({"tool_id": f"t{i}", "name": f"{w[0]}_{w[1]}_api_{i}", "description": f"Returns {w[0]} and {w[1]} data about {w[2]}.",
                      "category": w[0]})
    write_jsonl(out / "catalog.jsonl", tools)
    H = [t["tool_id"] for t in tools[:8]]
    json.dump(H, open(out / "heldout.json", "w"))
    tests, groups = [], ["G1_tool", "G1_category", "G1_instruction", "G2_category", "G2_instruction", "G3_instruction"]
    for i in range(120):
        t = tools[rng.randrange(40)]
        gold = [t["tool_id"]] + ([tools[rng.randrange(40)]["tool_id"]] if rng.random() < 0.15 else [])
        grp = "G1_tool" if t["tool_id"] in H else rng.choice(groups[2:])
        tests.append({"qid": f"q{i}", "query": f"please find {t['description'].split()[1]} {t['description'].split()[3]} for me", "gold": gold, "group": grp})
    write_jsonl(out / "tests.jsonl", tests)
    pool = []
    for i in range(600):
        t = tools[rng.randrange(40)]
        pool.append({"item_id": f"p{i}", "query": f"I need {t['description'].split()[1]} info {rng.choice(WORDS)}", "labels": [t["tool_id"]],
                     "meta_text": t["name"]})
    write_jsonl(out / "pool.jsonl", pool)
    for seed in (42, 123):
        r2 = random.Random(seed)
        for c in range(3):
            rows = []
            for _ in range(60):
                t = tools[r2.randrange(8, 40)]  # never a held-out tool
                rows.append({"tool_id": t["tool_id"], "text": f"use {t['name']} when asked about {t['description'].split()[1]} {r2.choice(WORDS)}", "kind": "scenario"})
            write_jsonl(out / f"clients/seed{seed}/client{c}.jsonl", rows)
        gen, pid = [], 0
        for t in tools:
            for k in range(5):
                gen.append({"tool_id": t["tool_id"], "style": "query", "text": f"can you get {t['description'].split()[1]} {WORDS[k]} details right now please", "pair_id": pid})
                gen.append({"tool_id": t["tool_id"], "style": "desc", "text": f"Returns {t['description'].split()[1]} {WORDS[k]} details for the given place and time", "pair_id": pid})
                pid += 1
        write_jsonl(out / f"gen/seed{seed}/gen.jsonl", gen)
    print("toy data in", out)


if __name__ == "__main__":
    main()
