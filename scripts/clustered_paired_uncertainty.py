#!/usr/bin/env python3
"""Dependence-aware paired uncertainty for two arms scored on the same queries.

Input: CSV/JSONL rows with:
    query_id, seed, tool_id, correct_a, correct_b

Reports accuracy difference A-B in percentage points, with query-clustered and
tool-clustered bootstrap intervals, per-seed differences, and exact McNemar.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    else:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    required = {"query_id", "seed", "tool_id", "correct_a", "correct_b"}
    missing = required - set(rows[0] if rows else {})
    if missing:
        raise SystemExit(f"missing required columns: {', '.join(sorted(missing))}")
    seen: set[tuple[str, str]] = set()
    for row in rows:
        row["query_id"] = str(row["query_id"])
        row["seed"] = str(row["seed"])
        row["tool_id"] = str(row["tool_id"])
        row["correct_a"] = int(row["correct_a"])
        row["correct_b"] = int(row["correct_b"])
        key = (row["query_id"], row["seed"])
        if key in seen:
            raise SystemExit(f"duplicate paired key query_id={row['query_id']!r} seed={row['seed']!r} in {path}")
        seen.add(key)
    return rows


def diff_pts(rows: list[dict[str, Any]]) -> float:
    n = len(rows)
    if n == 0:
        raise ValueError("cannot score an empty row set")
    a = sum(int(row["correct_a"]) for row in rows)
    b = sum(int(row["correct_b"]) for row in rows)
    return 100.0 * (a - b) / n


def cluster_boot(rows: list[dict[str, Any]], key: str, samples: int, rng: random.Random) -> tuple[float, float]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    ids = list(groups)
    out: list[float] = []
    for _ in range(samples):
        flat = [row for group_id in rng.choices(ids, k=len(ids)) for row in groups[group_id]]
        out.append(diff_pts(flat))
    out.sort()
    return out[int(0.025 * samples)], out[int(0.975 * samples) - 1]


def mcnemar(rows: list[dict[str, Any]]) -> tuple[int, int, float]:
    b = sum(1 for row in rows if row["correct_a"] == 1 and row["correct_b"] == 0)
    c = sum(1 for row in rows if row["correct_a"] == 0 and row["correct_b"] == 1)
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1))
    return b, c, min(1.0, 2.0 * p / (2**n))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("rows", type=Path)
    parser.add_argument("--margin", type=float, default=2.0)
    parser.add_argument("--B", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label", default="A-B")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = load_rows(args.rows)
    est = diff_pts(rows)
    ql, qh = cluster_boot(rows, "query_id", args.B, rng)
    tl, th = cluster_boot(rows, "tool_id", args.B, rng)
    def seed_sort_key(seed: str) -> tuple[int, int | str]:
        return (0, int(seed)) if seed.isdigit() else (1, seed)

    per_seed = {
        seed: diff_pts([row for row in rows if row["seed"] == seed])
        for seed in sorted({row["seed"] for row in rows}, key=seed_sort_key)
    }
    b, c, p = mcnemar(rows)
    wide = (min(ql, tl), max(qh, th))
    no_diff = wide[0] <= 0 <= wide[1]
    disregard = wide[0] >= -args.margin and wide[1] <= args.margin

    print(f"{args.label}: diff={est:+.2f} pts  n_rows={len(rows)}")
    print(f"  query-clustered 95% CI: [{ql:+.2f}, {qh:+.2f}]   tool-clustered: [{tl:+.2f}, {th:+.2f}]")
    print("  per-seed: " + ", ".join(f"{seed}:{delta:+.2f}" for seed, delta in per_seed.items()))
    print(f"  McNemar discordant A>B={b} B>A={c} exact two-sided p={p:.3g}")
    print(f"  no difference detected (wider CI contains 0): {no_diff}")
    print(f"  small enough to disregard (wider CI within +/-{args.margin}): {disregard}")

    output = args.rows.with_name(args.rows.name + ".clustered.json")
    with output.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "label": args.label,
                "diff_pts": est,
                "ci_query": [ql, qh],
                "ci_tool": [tl, th],
                "per_seed": per_seed,
                "mcnemar": {"a_gt_b": b, "b_gt_a": c, "p": p},
                "no_difference_detected": no_diff,
                "disregardable": disregard,
                "margin": args.margin,
                "B": args.B,
                "rows": str(args.rows),
            },
            handle,
            indent=1,
            sort_keys=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
