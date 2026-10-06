#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_ROOT = Path(os.environ.get("FEDRAG_CANONICAL_ROOT", str(REPO_ROOT))).resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_stabletoolbench_federated import (
    DEFAULT_STB_ROOT,
    DEFAULT_TOOLBENCH_INSTRUCTION_DIR,
    GROUPS,
    ProgressLogger,
    apply_junk_filter,
    assert_clean_tree,
    build_tool_registry,
    filter_eval_queries,
    filter_experience_items,
    load_stabletoolbench_queries,
    load_toolbench_training_items,
    save_json,
    stable_hash,
)
from scripts.run_stabletoolbench_heldout import filter_heldout_pool, sha256_texts


DEFAULT_OUTPUT_DIR = CANONICAL_ROOT / "artifacts" / "results" / "stabletoolbench_category_holdout_e7_inputs_r1"
DEFAULT_CATEGORY_GROUP = "G1_category"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E7 category-level hold-out exporter and pool-size gate.")
    parser.add_argument("--stb-root", type=Path, default=DEFAULT_STB_ROOT)
    parser.add_argument("--toolbench-instruction-dir", type=Path, default=DEFAULT_TOOLBENCH_INSTRUCTION_DIR)
    parser.add_argument("--groups", default=",".join(GROUPS))
    parser.add_argument("--category-source-group", default=DEFAULT_CATEGORY_GROUP)
    parser.add_argument("--min-pool-items", type=int, default=25000)
    parser.add_argument("--half-category-seed", type=int, default=20260927)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args()


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def categories_for_gold_tools(queries: list[Any], registry: dict[str, Any], source_group: str) -> list[str]:
    categories: set[str] = set()
    for query in queries:
        if query.group != source_group:
            continue
        for tool in query.gold_tools:
            doc = registry.get(tool)
            if doc is not None:
                categories.update(doc.categories)
    return sorted(categories)


def tools_in_categories(registry: dict[str, Any], categories: set[str]) -> list[str]:
    return sorted(tool for tool, doc in registry.items() if set(doc.categories) & categories)


def run_filter(items: list[Any], heldout_tools: list[str]) -> tuple[list[Any], dict[str, Any]]:
    kept, info = filter_heldout_pool(items, set(heldout_tools))
    info = {
        **info,
        "heldout_tool_count": len(heldout_tools),
        "heldout_tools_sha256": sha256_texts(heldout_tools),
        "pool_sha256": stable_hash(
            [{"query_id": item.query_id, "query": item.query, "gold_tools": item.gold_tools} for item in kept]
        ),
    }
    return kept, info


def choose_seeded_half(categories: list[str], seed: int) -> list[str]:
    rng = random.Random(seed)
    shuffled = list(categories)
    rng.shuffle(shuffled)
    keep_n = max(1, len(shuffled) // 2)
    return sorted(shuffled[:keep_n])


def summarize_eval_queries(queries: list[Any], heldout_tools: set[str]) -> dict[str, Any]:
    heldout = [query for query in queries if query.gold_tools and all(tool in heldout_tools for tool in query.gold_tools)]
    partial = [
        query
        for query in queries
        if query.gold_tools and any(tool in heldout_tools for tool in query.gold_tools) and not all(tool in heldout_tools for tool in query.gold_tools)
    ]
    by_group: dict[str, int] = {}
    for query in heldout:
        by_group[query.group] = by_group.get(query.group, 0) + 1
    return {
        "heldout_query_count": len(heldout),
        "partial_heldout_query_count": len(partial),
        "heldout_query_count_by_group": by_group,
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    commit, dirty = assert_clean_tree(allow_dirty=args.allow_dirty)
    progress = ProgressLogger(args.output_dir)
    progress.log("start", repo_commit=commit, dirty_entry_count=len(dirty), output_dir=str(args.output_dir))

    groups = parse_csv(args.groups)
    queries = load_stabletoolbench_queries(args.stb_root, groups)
    train_items = load_toolbench_training_items(args.toolbench_instruction_dir)
    registry = build_tool_registry(queries + train_items)
    registry, registry_filter = apply_junk_filter(registry)
    queries, query_filter = filter_eval_queries(queries, registry)
    train_items, train_filter = filter_experience_items(train_items, registry)
    progress.log(
        "data_loaded",
        query_count=len(queries),
        train_count=len(train_items),
        registry_tool_count=len(registry),
        **registry_filter,
        **query_filter,
        **train_filter,
    )

    source_categories = categories_for_gold_tools(queries, registry, args.category_source_group)
    if not source_categories:
        raise RuntimeError(f"no categories found for source group {args.category_source_group}")

    all_tools = tools_in_categories(registry, set(source_categories))
    all_kept, all_info = run_filter(train_items, all_tools)
    progress.log(
        "category_filter_candidate",
        mode="all_categories",
        heldout_category_count=len(source_categories),
        **all_info,
    )

    selected_categories = source_categories
    selected_tools = all_tools
    selected_kept = all_kept
    selected_info = all_info
    mode = "all_categories"
    if len(all_kept) < args.min_pool_items:
        selected_categories = choose_seeded_half(source_categories, args.half_category_seed)
        selected_tools = tools_in_categories(registry, set(selected_categories))
        selected_kept, selected_info = run_filter(train_items, selected_tools)
        mode = "seeded_half_categories"
        progress.log(
            "category_filter_candidate",
            mode=mode,
            half_category_seed=args.half_category_seed,
            heldout_category_count=len(selected_categories),
            **selected_info,
        )

    if len(selected_kept) < args.min_pool_items:
        raise RuntimeError(
            f"E7 pool-size gate failed: {len(selected_kept)} items remain, need at least {args.min_pool_items}"
        )
    if selected_info["remaining_items_with_heldout_label"] != 0:
        raise RuntimeError("E7 held-out label filter failed")

    selected_heldout_set = set(selected_tools)
    eval_summary = summarize_eval_queries(queries, selected_heldout_set)
    summary = {
        "paper_eligible": True,
        "repo_commit": commit,
        "dirty_entry_count": len(dirty),
        "mode": mode,
        "category_source_group": args.category_source_group,
        "min_pool_items": args.min_pool_items,
        "half_category_seed": args.half_category_seed,
        "requested_category_count": len(source_categories),
        "requested_categories": source_categories,
        "requested_categories_sha256": sha256_texts(source_categories),
        "heldout_category_count": len(selected_categories),
        "heldout_categories": selected_categories,
        "heldout_categories_sha256": sha256_texts(selected_categories),
        "heldout_tool_count": len(selected_tools),
        "heldout_tools": selected_tools,
        "heldout_tools_sha256": sha256_texts(selected_tools),
        "remaining_pool_count": len(selected_kept),
        "pool_size_gate_passed": True,
        "heldout_filter": selected_info,
        "eval_summary": eval_summary,
        "junk_filter": {**registry_filter, **query_filter, **train_filter},
    }
    save_json(args.output_dir / "summary.json", summary)
    save_json(args.output_dir / "heldout_categories.json", selected_categories)
    save_json(args.output_dir / "heldout_tools.json", selected_tools)
    progress.log("complete", **{k: v for k, v in summary.items() if k not in {"heldout_tools", "heldout_categories", "requested_categories"}})
    print(
        json.dumps(
            {
                "mode": mode,
                "heldout_category_count": len(selected_categories),
                "heldout_tool_count": len(selected_tools),
                "remaining_pool_count": len(selected_kept),
                "eval_summary": eval_summary,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
