#!/usr/bin/env python3
"""D5 paired tests from completed row-level artifacts.

Each comparison is paired by query id and reports:
- exact McNemar test for binary paired outcomes,
- paired bootstrap CI for B-A rate difference,
- paired TOST at a pre-specified ±2 point margin.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rows(path: Path, subset: str | None = None, outcome: str = "gold_in_top_k") -> dict[str, bool]:
    data = json.loads(path.read_text())
    rows = data.get("rows") or []
    out: dict[str, bool] = {}
    for row in rows:
        if subset and str(row.get("subset")) != subset:
            continue
        qid = str(row.get("row_id") or row.get("query_id"))
        if qid in out:
            raise ValueError(f"duplicate paired key {qid!r} in {path} subset={subset}")
        if outcome == "gold_in_top_k":
            val = bool(row.get("gold_in_top_k", row.get("gold_in_top_5")))
        elif outcome == "correct":
            val = bool(row.get("correct", row.get("routed_correctly")))
        elif outcome == "top1_correct":
            val = bool(row.get("top1_correct", row.get("retrieval_top1_correct")))
        else:
            raise ValueError(outcome)
        out[qid] = val
    return out


def paired_values(a: dict[str, bool], b: dict[str, bool]) -> tuple[list[str], list[int]]:
    keys = sorted(set(a) & set(b))
    diffs = [(1 if b[key] else 0) - (1 if a[key] else 0) for key in keys]
    return keys, diffs


def mcnemar(a: dict[str, bool], b: dict[str, bool]) -> dict[str, Any]:
    keys = sorted(set(a) & set(b))
    b01 = sum((not a[k]) and b[k] for k in keys)
    b10 = sum(a[k] and (not b[k]) for k in keys)
    discordant = b01 + b10
    if discordant == 0:
        p = 1.0
        stat = 0.0
    else:
        tail = sum(math.comb(discordant, i) for i in range(0, min(b01, b10) + 1)) / (2**discordant)
        p = min(1.0, 2 * tail)
        stat = (abs(b01 - b10) - 1) ** 2 / discordant
    a_rate = sum(a[k] for k in keys) / len(keys) if keys else 0.0
    b_rate = sum(b[k] for k in keys) / len(keys) if keys else 0.0
    return {
        "n": len(keys),
        "a_only": b10,
        "b_only": b01,
        "discordant": discordant,
        "p_exact_two_sided": p,
        "chi2_cc": stat,
        "a_rate": a_rate,
        "b_rate": b_rate,
        "b_minus_a": b_rate - a_rate,
    }


def bootstrap_ci(diff: list[int], *, reps: int = 5000, alpha: float = 0.05, seed: int = 20260911) -> dict[str, Any]:
    if not diff:
        return {"n": 0, "mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "reps": reps, "alpha": alpha}
    rng = random.Random(seed)
    n = len(diff)
    means = []
    for _ in range(reps):
        means.append(sum(diff[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[max(0, int((alpha / 2) * reps))]
    hi = means[min(reps - 1, int((1 - alpha / 2) * reps))]
    return {"n": n, "mean": sum(diff) / n, "ci_low": lo, "ci_high": hi, "reps": reps, "alpha": alpha}


def paired_tost(diff: list[int], *, margin: float = 0.02, alpha: float = 0.05) -> dict[str, Any]:
    n = len(diff)
    if n < 2:
        return {"n": n, "margin": margin, "equivalent": False, "reason": "need at least two paired rows"}
    mean = sum(diff) / n
    sd = statistics.stdev(diff)
    if sd == 0.0:
        equivalent = -margin < mean < margin
        return {
            "n": n,
            "mean": mean,
            "sd": sd,
            "margin": margin,
            "p_lower": 0.0 if mean > -margin else 1.0,
            "p_upper": 0.0 if mean < margin else 1.0,
            "p_tost": 0.0 if equivalent else 1.0,
            "equivalent": equivalent,
            "method": "degenerate_exact",
        }
    se = sd / math.sqrt(n)
    norm_cdf = lambda z: 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    z_lower = (mean + margin) / se
    z_upper = (mean - margin) / se
    p_lower = 1.0 - norm_cdf(z_lower)
    p_upper = norm_cdf(z_upper)
    p_tost = max(p_lower, p_upper)
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "se": se,
        "margin": margin,
        "alpha": alpha,
        "p_lower": p_lower,
        "p_upper": p_upper,
        "p_tost": p_tost,
        "equivalent": p_lower < alpha and p_upper < alpha,
        "method": "paired_normal_approx",
    }


def add_test(
    tests: list[dict[str, Any]],
    name: str,
    a_path: Path,
    b_path: Path,
    *,
    subset: str | None,
    outcome: str,
    a_label: str,
    b_label: str,
) -> None:
    if not a_path.exists() or not b_path.exists():
        tests.append({"name": name, "blocked": True, "missing": [str(p) for p in (a_path, b_path) if not p.exists()]})
        return
    a = load_rows(a_path, subset, outcome)
    b = load_rows(b_path, subset, outcome)
    keys, diff = paired_values(a, b)
    tests.append(
        {
            "name": name,
            "subset": subset,
            "outcome": outcome,
            "a_label": a_label,
            "b_label": b_label,
            "a_path": str(a_path),
            "b_path": str(b_path),
            "a_sha256": sha256_path(a_path),
            "b_sha256": sha256_path(b_path),
            "paired_key_count": len(keys),
            "mcnemar": mcnemar(a, b),
            "bootstrap_diff_ci": bootstrap_ci(diff),
            "tost_margin_0_02": paired_tost(diff, margin=0.02),
        }
    )


def seed_e6_root(seed: int) -> Path:
    if seed == 42:
        return Path("artifacts/results/stabletoolbench_heldout_retriever_compare_e6_r4")
    return Path(f"artifacts/results/stabletoolbench_heldout_retriever_compare_e6_seed{seed}_r4")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("artifacts/verification/d5_paired_tests_r2/summary.json"))
    args = ap.parse_args()
    tests: list[dict[str, Any]] = []

    for seed in (42, 123, 456):
        root = seed_e6_root(seed)
        for retriever in ("jina", "bm25", "bge"):
            add_test(tests, f"e6_seed{seed}_{retriever}_heldout_recall_shared_vs_docs", root / f"seed_{seed}/{retriever}/docs_only.json", root / f"seed_{seed}/{retriever}/synapse_shared.json", subset="heldout", outcome="gold_in_top_k", a_label="docs_only", b_label="synapse_shared")
            add_test(tests, f"e6_seed{seed}_{retriever}_labeled_recall_shared_vs_docs", root / f"seed_{seed}/{retriever}/docs_only.json", root / f"seed_{seed}/{retriever}/synapse_shared.json", subset="labeled", outcome="gold_in_top_k", a_label="docs_only", b_label="synapse_shared")

    e8 = Path("artifacts/results/stabletoolbench_two_index_router_e8_r1")
    e8_budget = Path("artifacts/results/stabletoolbench_two_index_router_e8_budgeted_distinct5_r2")
    e8b = Path("artifacts/results/stabletoolbench_two_index_router_e8_budgeted_retrievers_r2")
    for seed in (42, 123, 456):
        add_test(tests, f"e8_seed{seed}_jina_rrf_vs_docs_heldout_recall", seed_e6_root(seed) / f"seed_{seed}/jina/docs_only.json", e8 / f"seed_{seed}/jina/union_rrf.json", subset="heldout", outcome="gold_in_top_k", a_label="docs_only", b_label="union_rrf")
        add_test(tests, f"e8_seed{seed}_jina_budget_vs_docs_heldout_recall", seed_e6_root(seed) / f"seed_{seed}/jina/docs_only.json", e8_budget / f"seed_{seed}/jina/docs_plus_experience_backfill.json", subset="heldout", outcome="gold_in_top_k", a_label="docs_only", b_label="budgeted")
        add_test(tests, f"e8_seed{seed}_jina_budget_vs_docs_heldout_accuracy", seed_e6_root(seed) / f"seed_{seed}/jina/docs_only.json", e8_budget / f"seed_{seed}/jina/docs_plus_experience_backfill.json", subset="heldout", outcome="correct", a_label="docs_only", b_label="budgeted")
        add_test(tests, f"e8_seed{seed}_jina_budget_vs_docs_labeled_accuracy", seed_e6_root(seed) / f"seed_{seed}/jina/docs_only.json", e8_budget / f"seed_{seed}/jina/docs_plus_experience_backfill.json", subset="labeled", outcome="correct", a_label="docs_only", b_label="budgeted")
        add_test(tests, f"e8_seed{seed}_jina_budget_vs_docs_overall_accuracy", seed_e6_root(seed) / f"seed_{seed}/jina/docs_only.json", e8_budget / f"seed_{seed}/jina/docs_plus_experience_backfill.json", subset=None, outcome="correct", a_label="docs_only", b_label="budgeted")
        for retriever in ("bm25", "bge"):
            add_test(tests, f"e8_seed{seed}_{retriever}_budget_vs_docs_heldout_recall", seed_e6_root(seed) / f"seed_{seed}/{retriever}/docs_only.json", e8b / f"seed_{seed}/{retriever}/docs_plus_experience_backfill.json", subset="heldout", outcome="gold_in_top_k", a_label="docs_only", b_label="budgeted")
            add_test(tests, f"e8_seed{seed}_{retriever}_budget_vs_docs_heldout_accuracy", seed_e6_root(seed) / f"seed_{seed}/{retriever}/docs_only.json", e8b / f"seed_{seed}/{retriever}/docs_plus_experience_backfill.json", subset="heldout", outcome="correct", a_label="docs_only", b_label="budgeted")
            add_test(tests, f"e8_seed{seed}_{retriever}_budget_vs_docs_labeled_accuracy", seed_e6_root(seed) / f"seed_{seed}/{retriever}/docs_only.json", e8b / f"seed_{seed}/{retriever}/docs_plus_experience_backfill.json", subset="labeled", outcome="correct", a_label="docs_only", b_label="budgeted")
            add_test(tests, f"e8_seed{seed}_{retriever}_budget_vs_docs_overall_accuracy", seed_e6_root(seed) / f"seed_{seed}/{retriever}/docs_only.json", e8b / f"seed_{seed}/{retriever}/docs_plus_experience_backfill.json", subset=None, outcome="correct", a_label="docs_only", b_label="budgeted")

    sparse = Path("artifacts/results/toolret_sparse_heldout_e7_r1")
    for seed in (42, 123, 456):
        for retriever in ("jina", "bm25", "bge"):
            add_test(tests, f"e7_sparse_seed{seed}_{retriever}_heldout_recall_shared_vs_docs", sparse / f"seed_{seed}/{retriever}/docs_only.json", sparse / f"seed_{seed}/{retriever}/shared.json", subset="heldout", outcome="gold_in_top_k", a_label="docs_only", b_label="shared")
            add_test(tests, f"e7_sparse_seed{seed}_{retriever}_labeled_recall_shared_vs_docs", sparse / f"seed_{seed}/{retriever}/docs_only.json", sparse / f"seed_{seed}/{retriever}/shared.json", subset="labeled", outcome="gold_in_top_k", a_label="docs_only", b_label="shared")

    for rate in (0, 60):
        for seed in (42, 123, 456):
            add_test(tests, f"table4_subset_rate{rate:02d}_concat_vs_synapse_seed{seed}", Path(f"artifacts/results/stabletoolbench_typed_conflictlog_distinct5_table4_r1/conflict_{rate:02d}/seed_{seed}.json"), Path(f"artifacts/results/stabletoolbench_concat_control_distinct5_r2/subset_g1g2_instruction/conflict_{rate:02d}/seed_{seed}.json"), subset=None, outcome="correct", a_label="synapse", b_label="concat")
            add_test(tests, f"table3_full_concat_vs_synapse_seed{seed}", Path(f"artifacts/verification/stabletoolbench_benchmark_cached_replay_r2/seed_{seed}/synapse.json"), Path(f"artifacts/results/stabletoolbench_concat_control_distinct5_r2/full/conflict_00/seed_{seed}.json"), subset=None, outcome="correct", a_label="synapse", b_label="concat")

    frozen_runs = {
        "llama31_8b_seed42": Path("artifacts/results/stabletoolbench_frozen_renderswap_seed42_r1"),
        "llama31_8b_seed123": Path("artifacts/results/stabletoolbench_frozen_renderswap_seed123_r1"),
        "llama31_8b_seed456": Path("artifacts/results/stabletoolbench_frozen_renderswap_seed456_r1"),
        "qwen25_3b_seed42": Path("artifacts/results/stabletoolbench_frozen_renderswap_qwen25_3b_seeds42_123_e2x_r5"),
        "qwen25_3b_seed123": Path("artifacts/results/stabletoolbench_frozen_renderswap_qwen25_3b_seeds42_123_e2x_r5"),
        "qwen25_3b_seed456": Path("artifacts/results/stabletoolbench_frozen_renderswap_qwen25_3b_seed456_e2x_r1"),
        "qwen25_7b_seed42": Path("artifacts/results/stabletoolbench_frozen_renderswap_qwen25_7b_seeds42_123_e2x_r1"),
        "qwen25_7b_seed123": Path("artifacts/results/stabletoolbench_frozen_renderswap_qwen25_7b_seeds42_123_e2x_r1"),
        "qwen25_7b_seed456": Path("artifacts/results/stabletoolbench_frozen_renderswap_qwen25_7b_seed456_r1"),
    }
    for label, root in frozen_runs.items():
        seed = int(label.rsplit("seed", 1)[1])
        for merge in ("typed_conflictlog", "flat_majority"):
            for rate in (0, 60):
                add_test(tests, f"frozen_swap_{label}_{merge}_rate{rate:02d}_flat_vs_typed_render", root / merge / f"conflict_{rate:02d}" / "typed" / f"seed_{seed}.json", root / merge / f"conflict_{rate:02d}" / "flat" / f"seed_{seed}.json", subset=None, outcome="correct", a_label="typed_render", b_label="flat_render")

    conflict_roots = {
        42: Path("artifacts/results/stabletoolbench_2x2_seed42_r3b"),
        456: Path("artifacts/results/stabletoolbench_2x2_seed456_r2"),
        7: Path("artifacts/results/stabletoolbench_2x2_extra_seeds_r3"),
        1234: Path("artifacts/results/stabletoolbench_2x2_extra_seeds_r3"),
    }
    for seed, root in conflict_roots.items():
        for rate in (0, 60):
            base = root / "typed_conflictlog" / f"conflict_{rate:02d}" / f"seed_{seed}.json"
            for other in ("typed_majority", "typed_round_delayed"):
                add_test(tests, f"conflict_policy_seed{seed}_rate{rate:02d}_{other}_vs_conflictlog", base, root / other / f"conflict_{rate:02d}" / f"seed_{seed}.json", subset=None, outcome="correct", a_label="typed_conflictlog", b_label=other)

    out = {"tests": tests}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    for test in tests:
        if test.get("blocked"):
            print("BLOCKED", test["name"], test["missing"])
            continue
        m = test["mcnemar"]
        t = test["tost_margin_0_02"]
        if any(key in test["name"] for key in ("heldout", "concat", "frozen_swap", "conflict_policy")):
            print(f"{test['name']}: {test['a_label']}={m['a_rate']:.3f} {test['b_label']}={m['b_rate']:.3f} diff={m['b_minus_a']:.3f} p={m['p_exact_two_sided']:.3g} tost_equiv={t.get('equivalent')}")
    print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
