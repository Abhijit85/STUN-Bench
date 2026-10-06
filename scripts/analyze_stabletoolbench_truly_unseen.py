#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def stable_hash(payload: Any) -> str:
    return __import__("hashlib").sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
    ).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze StableToolBench covered vs truly-uncovered subsets from clean A1 artifacts."
    )
    parser.add_argument(
        "--filtered-pool",
        type=Path,
        default=Path("artifacts/verification/stabletoolbench_query_classifier_clean_r1/filtered_pool.json"),
    )
    parser.add_argument(
        "--a1-roots",
        type=Path,
        nargs="+",
        default=[
            Path("artifacts/verification/stabletoolbench_a1_seed42_eval_r16"),
            Path("artifacts/verification/stabletoolbench_a1_seed123_eval_r16"),
            Path("artifacts/verification/stabletoolbench_a1_seed456_eval_r16"),
        ],
    )
    parser.add_argument(
        "--classifier-roots",
        type=Path,
        nargs="+",
        default=[Path("artifacts/verification/stabletoolbench_query_classifier_clean_r1")],
    )
    parser.add_argument("--client-count", type=int, default=5)
    parser.add_argument("--partition-mode", type=str, default="category", choices=["category", "iid"])
    parser.add_argument("--max-items-per-client", type=int, default=5000)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("artifacts/verification/stabletoolbench_truly_unseen_analysis.json"),
    )
    return parser.parse_args()


def parse_seed_from_root(path: Path) -> int:
    import re

    match = re.search(r"seed(\d+)", path.name)
    if match:
        return int(match.group(1))
    for part in path.name.split("_"):
        if part.isdigit():
            return int(part)
    raise ValueError(f"Could not parse seed from {path}")


def limit_experience_items(items: list[dict[str, Any]], max_items: int, seed: int) -> list[dict[str, Any]]:
    if max_items <= 0 or len(items) <= max_items:
        return list(items)
    sampled = list(items)
    rng = random.Random(seed)
    rng.shuffle(sampled)
    return sampled[:max_items]


def assign_clients(items: list[dict[str, Any]], client_count: int, partition_mode: str, seed: int) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    clients = {f"client_{idx}": [] for idx in range(client_count)}
    if partition_mode == "iid":
        shuffled = list(items)
        rng.shuffle(shuffled)
        for idx, item in enumerate(shuffled):
            clients[f"client_{idx % client_count}"].append(item)
        return clients
    categories = sorted({str(item["primary_category"]) for item in items})
    rng.shuffle(categories)
    category_to_client = {category: f"client_{idx % client_count}" for idx, category in enumerate(categories)}
    for item in items:
        clients[category_to_client[str(item["primary_category"])]] .append(item)
    return clients


def limit_client_items(clients: dict[str, list[dict[str, Any]]], max_items_per_client: int, seed: int) -> dict[str, list[dict[str, Any]]]:
    if max_items_per_client <= 0:
        return {client_id: list(items) for client_id, items in clients.items()}
    limited: dict[str, list[dict[str, Any]]] = {}
    for offset, client_id in enumerate(sorted(clients)):
        limited[client_id] = limit_experience_items(clients[client_id], max_items_per_client, seed + offset + 1)
    return limited


def fit_pool_sha(items: list[dict[str, Any]]) -> str:
    return stable_hash(
        [
            {
                "query_id": item["query_id"],
                "query": item["query"],
                "group": item["group"],
                "gold_tools": list(item["gold_tools"]),
                "primary_category": item["primary_category"],
            }
            for item in items
        ]
    )


def load_seed_rows(root: Path, seed: int) -> dict[str, dict[str, Any]]:
    seed_dir = root / f"seed_{seed}"
    rows_by_arm: dict[str, dict[str, Any]] = {}
    for arm in ("synapse", "centralized", "flat_pool", "local_only"):
        payload = json.loads((seed_dir / f"{arm}.json").read_text(encoding="utf-8"))
        rows_by_arm[arm] = payload
    return rows_by_arm


def per_subset_accuracy(rows: list[dict[str, Any]], subset_ids: set[str]) -> float | None:
    bucket = [row for row in rows if str(row["query_id"]) in subset_ids]
    if not bucket:
        return None
    return sum(1 for row in bucket if row["routed_correctly"]) / len(bucket)


def mean_sd(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "sd": 0.0}
    return {
        "mean": statistics.mean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def main() -> None:
    args = parse_args()
    filtered = json.loads(args.filtered_pool.read_text(encoding="utf-8"))
    pool_items = filtered["items"]
    pool_label_set = {tool for item in pool_items for tool in item["gold_tools"]}

    classifier_root = args.classifier_roots[0]
    classifier_summary = json.loads((classifier_root / "combined_summary.json").read_text(encoding="utf-8"))

    per_seed: dict[str, Any] = {}
    uncovered_counts: list[float] = []
    covered_counts: list[float] = []
    arm_uncovered: dict[str, list[float]] = defaultdict(list)
    arm_covered: dict[str, list[float]] = defaultdict(list)

    g1_tool_gold_tools: set[str] = set()
    g1_category_gold_tools: set[str] = set()
    g1_tool_query_covered_in_pool = None
    g1_category_query_covered_in_pool = None

    for a1_root in args.a1_roots:
        seed = parse_seed_from_root(a1_root)
        rows_by_arm = load_seed_rows(a1_root, seed)
        synapse_rows = rows_by_arm["synapse"]["rows"]
        eval_rows = synapse_rows

        clients = assign_clients(pool_items, args.client_count, args.partition_mode, seed)
        clients = limit_client_items(clients, args.max_items_per_client, seed)
        fit_items = [item for client_id in sorted(clients) for item in clients[client_id]]
        fit_labels = {tool for item in fit_items for tool in item["gold_tools"]}
        fit_sha = fit_pool_sha(fit_items)

        qc_seed = json.loads((classifier_root / f"seed_{seed}.json").read_text(encoding="utf-8"))
        rows_by_arm["query_classifier"] = qc_seed
        classifier_fit_sha = qc_seed["classifier_fit_pool_sha256"]

        uncovered_ids: set[str] = set()
        covered_ids: set[str] = set()
        seed_g1_tool_query_covered_in_pool = 0
        seed_g1_category_query_covered_in_pool = 0
        for row in eval_rows:
            golds = set(map(str, row["gold_parent_tools"]))
            if golds & fit_labels:
                covered_ids.add(str(row["query_id"]))
            else:
                uncovered_ids.add(str(row["query_id"]))

            if row["group"] == "G1_tool":
                g1_tool_gold_tools.update(golds)
                if golds & pool_label_set:
                    seed_g1_tool_query_covered_in_pool += 1
            if row["group"] == "G1_category":
                g1_category_gold_tools.update(golds)
                if golds & pool_label_set:
                    seed_g1_category_query_covered_in_pool += 1

        if g1_tool_query_covered_in_pool is None:
            g1_tool_query_covered_in_pool = seed_g1_tool_query_covered_in_pool
        if g1_category_query_covered_in_pool is None:
            g1_category_query_covered_in_pool = seed_g1_category_query_covered_in_pool

        seed_result = {
            "seed": seed,
            "classifier_fit_pool_sha256_recomputed": fit_sha,
            "classifier_fit_pool_sha256_recorded": classifier_fit_sha,
            "classifier_fit_sha_match": fit_sha == classifier_fit_sha,
            "fit_label_count": len(fit_labels),
            "n_uncovered": len(uncovered_ids),
            "n_covered": len(covered_ids),
            "accuracies_uncovered": {},
            "accuracies_covered": {},
        }

        for arm_name, payload in rows_by_arm.items():
            uncovered_acc = per_subset_accuracy(payload["rows"], uncovered_ids)
            covered_acc = per_subset_accuracy(payload["rows"], covered_ids)
            seed_result["accuracies_uncovered"][arm_name] = uncovered_acc
            seed_result["accuracies_covered"][arm_name] = covered_acc
            if uncovered_acc is not None:
                arm_uncovered[arm_name].append(uncovered_acc)
            if covered_acc is not None:
                arm_covered[arm_name].append(covered_acc)

        uncovered_counts.append(len(uncovered_ids))
        covered_counts.append(len(covered_ids))
        per_seed[str(seed)] = seed_result

    g1_tool_unique_in_pool = sum(1 for tool in g1_tool_gold_tools if tool in pool_label_set)
    g1_category_unique_in_pool = sum(1 for tool in g1_category_gold_tools if tool in pool_label_set)

    summary = {
        "paper_eligible": True,
        "filtered_pool_path": str(args.filtered_pool),
        "filtered_pool_count": len(pool_items),
        "filtered_pool_sha256": filtered["filtered_pool_sha256"],
        "classifier_summary_path": str(classifier_root / "combined_summary.json"),
        "classifier_overall_accuracy": classifier_summary["query_classifier"]["mean_accuracy"],
        "per_seed": per_seed,
        "aggregate": {
            "n_uncovered": mean_sd(uncovered_counts),
            "n_covered": mean_sd(covered_counts),
            "accuracies_uncovered": {arm: mean_sd(values) for arm, values in sorted(arm_uncovered.items())},
            "accuracies_covered": {arm: mean_sd(values) for arm, values in sorted(arm_covered.items())},
        },
        "diagnosis": {
            "g1_tool_query_count": 152,
            "g1_tool_unique_gold_tools": len(g1_tool_gold_tools),
            "g1_tool_unique_gold_tools_in_filtered_pool": g1_tool_unique_in_pool,
            "g1_tool_query_count_with_pool_label": g1_tool_query_covered_in_pool,
            "g1_category_query_count": 134,
            "g1_category_unique_gold_tools": len(g1_category_gold_tools),
            "g1_category_unique_gold_tools_in_filtered_pool": g1_category_unique_in_pool,
            "g1_category_query_count_with_pool_label": g1_category_query_covered_in_pool,
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"output_json={args.output_json}")
    print(f"tu_n_mean={summary['aggregate']['n_uncovered']['mean']:.3f}")
    print(f"tu_n_sd={summary['aggregate']['n_uncovered']['sd']:.3f}")
    for arm in ("synapse", "centralized", "flat_pool", "local_only"):
        stats = summary["aggregate"]["accuracies_uncovered"].get(arm, {"mean": 0.0, "sd": 0.0})
        print(f"uncovered_{arm}={stats['mean']:.6f} sd={stats['sd']:.6f}")
    for arm in ("synapse", "centralized", "flat_pool", "local_only"):
        stats = summary["aggregate"]["accuracies_covered"].get(arm, {"mean": 0.0, "sd": 0.0})
        print(f"covered_{arm}={stats['mean']:.6f} sd={stats['sd']:.6f}")
    print(
        "g1_tool_pool_coverage="
        f"{summary['diagnosis']['g1_tool_unique_gold_tools_in_filtered_pool']}/"
        f"{summary['diagnosis']['g1_tool_unique_gold_tools']}"
    )
    print(
        "g1_category_pool_coverage="
        f"{summary['diagnosis']['g1_category_unique_gold_tools_in_filtered_pool']}/"
        f"{summary['diagnosis']['g1_category_unique_gold_tools']}"
    )


if __name__ == "__main__":
    main()
