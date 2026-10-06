#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_ROOT = Path(os.environ.get("FEDRAG_CANONICAL_ROOT", str(REPO_ROOT))).resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from math_qa import JinaAIClient
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
    remove_near_duplicate_eval_overlaps,
    resolve_local_embedder,
    save_json,
    stable_hash,
    temporary_env,
)
from scripts.run_stabletoolbench_heldout import (
    HELDOUT_GROUPS,
    heldout_tool_set,
    item_mentions_heldout_tool,
    sha256_texts,
)


DEFAULT_OUTPUT_DIR = CANONICAL_ROOT / "artifacts" / "release" / "stabletoolbench_filters_r1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export per-item StableToolBench filter/removal reasons keyed by unique row_id."
    )
    parser.add_argument("--stb-root", type=Path, default=DEFAULT_STB_ROOT)
    parser.add_argument("--toolbench-instruction-dir", type=Path, default=DEFAULT_TOOLBENCH_INSTRUCTION_DIR)
    parser.add_argument("--groups", default=",".join(GROUPS))
    parser.add_argument("--embed-model", default="jina-embeddings-v2-base-en")
    parser.add_argument("--pool-embedding-cache-dir", type=Path, default=CANONICAL_ROOT / "artifacts" / "cache" / "stabletoolbench")
    parser.add_argument("--contamination-near-duplicate-threshold", type=float, default=0.95)
    parser.add_argument("--contamination-report-threshold", type=float, default=0.90)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args()


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def row_id_for_item(index: int, item: Any) -> str:
    # ToolBench query_id is not unique; include stable load order and content hash.
    digest = stable_hash(
        {
            "index": index,
            "query_id": item.query_id,
            "query": item.query,
            "gold_tools": item.gold_tools,
            "group": item.group,
        }
    )[:16]
    return f"{item.group}:{index:06d}:{digest}"


def item_record(row_id: str, item: Any) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "query_id": item.query_id,
        "group": item.group,
        "query": item.query,
        "gold_tools": list(item.gold_tools),
        "primary_category": item.primary_category,
        "categories": list(item.categories),
    }


def item_signature(item: Any) -> str:
    # Filter steps copy rows and may prune gold_tools, so only use source-stable fields here.
    return stable_hash(
        {
            "query_id": item.query_id,
            "query": item.query,
            "group": item.group,
        }
    )


def row_id_lookup(raw_items: list[Any], row_ids: list[str]) -> dict[str, deque[str]]:
    lookup: dict[str, deque[str]] = defaultdict(deque)
    for item, row_id in zip(raw_items, row_ids):
        lookup[item_signature(item)].append(row_id)
    return lookup


def row_ids_for_items(items: list[Any], raw_lookup: dict[str, deque[str]]) -> set[str]:
    lookup = {key: deque(values) for key, values in raw_lookup.items()}
    out: set[str] = set()
    for item in items:
        signature = item_signature(item)
        if not lookup.get(signature):
            raise RuntimeError(f"could not map filtered item to a release row_id: {item.query_id}")
        out.add(lookup[signature].popleft())
    return out


def exact_eval_texts(queries: list[Any]) -> set[str]:
    return {query.query.strip().lower() for query in queries if query.query.strip()}


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    commit, dirty = assert_clean_tree(allow_dirty=args.allow_dirty)
    progress = ProgressLogger(args.output_dir)
    progress.log("start", repo_commit=commit, dirty_entry_count=len(dirty), output_dir=str(args.output_dir))

    groups = parse_csv(args.groups)
    queries = load_stabletoolbench_queries(args.stb_root, groups)
    raw_items = load_toolbench_training_items(args.toolbench_instruction_dir)
    row_ids = [row_id_for_item(idx, item) for idx, item in enumerate(raw_items)]
    if len(row_ids) != len(set(row_ids)):
        raise RuntimeError("row_id uniqueness assertion failed")
    raw_lookup = row_id_lookup(raw_items, row_ids)

    registry = build_tool_registry(queries + raw_items)
    registry, registry_filter = apply_junk_filter(registry)
    queries, query_filter = filter_eval_queries(queries, registry)
    junk_kept, train_filter = filter_experience_items(raw_items, registry)
    junk_kept_row_ids = row_ids_for_items(junk_kept, raw_lookup)
    progress.log(
        "junk_filter_done",
        raw_train_count=len(raw_items),
        junk_kept_count=len(junk_kept),
        query_count=len(queries),
        registry_tool_count=len(registry),
        **registry_filter,
        **query_filter,
        **train_filter,
    )

    heldout_tools, heldout_sanity = heldout_tool_set(queries, HELDOUT_GROUPS)
    heldout_set = set(heldout_tools)
    exact_texts = exact_eval_texts(queries)
    after_exact = [item for item in junk_kept if item.query.strip().lower() not in exact_texts]
    after_exact_row_ids = row_ids_for_items(after_exact, raw_lookup)
    exact_removed_row_ids = junk_kept_row_ids - after_exact_row_ids

    embedder_info = resolve_local_embedder()
    jina = JinaAIClient(api_keys=[])
    local_env = {
        "JINA_LOCAL_EMBED_MODEL": embedder_info["model_path"],
        "JINA_LOCAL_EMBED_DEVICE": embedder_info["device"],
        "JINA_LOCAL_EMBED_LOCAL_ONLY": embedder_info["local_only"],
        "JINA_API_KEY": None,
    }
    with temporary_env(local_env):
        after_near, near_info, _ = remove_near_duplicate_eval_overlaps(
            after_exact,
            queries,
            jina,
            args.embed_model,
            removal_threshold=args.contamination_near_duplicate_threshold,
            report_threshold=args.contamination_report_threshold,
            pool_embedding_cache_dir=args.pool_embedding_cache_dir,
            progress=progress,
        )
    after_near_row_ids = row_ids_for_items(after_near, raw_lookup)
    near_removed_row_ids = after_exact_row_ids - after_near_row_ids

    manifest_path = args.output_dir / "filter_items.jsonl"
    kept_count = 0
    reason_counts: dict[str, int] = {}
    with manifest_path.open("w", encoding="utf-8") as handle:
        for idx, item in enumerate(raw_items):
            row_id = row_ids[idx]
            reasons: list[str] = []
            if row_id not in junk_kept_row_ids:
                reasons.append("junk_tool_filter")
            elif row_id in exact_removed_row_ids:
                reasons.append("leak_exact_eval_query")
            elif row_id in near_removed_row_ids:
                reasons.append(f"leak_near_duplicate_eval_query_cos_ge_{args.contamination_near_duplicate_threshold:g}")
            else:
                if any(tool in heldout_set for tool in item.gold_tools):
                    reasons.append("heldout_label")
                if item_mentions_heldout_tool(item, heldout_set):
                    reasons.append("heldout_mention")
            if not reasons:
                kept_count += 1
            for reason in reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            handle.write(json.dumps({**item_record(row_id, item), "removal_reasons": reasons}, sort_keys=True) + "\n")

    summary = {
        "paper_eligible": True,
        "repo_commit": commit,
        "dirty_entry_count": len(dirty),
        "row_key": "row_id",
        "row_id_uniqueness_assertion": True,
        "raw_train_count": len(raw_items),
        "release_manifest_path": str(manifest_path),
        "release_manifest_sha256": stable_hash(manifest_path.read_text(encoding="utf-8").splitlines()),
        "kept_after_all_filters": kept_count,
        "removal_reason_counts": reason_counts,
        "heldout_tools_sha256": sha256_texts(heldout_tools),
        "heldout_sanity": heldout_sanity,
        "junk_filter": {**registry_filter, **query_filter, **train_filter},
        "near_duplicate_filter": {k: v for k, v in near_info.items() if k != "pool_items_removed_near_dup_query_ids"},
    }
    save_json(args.output_dir / "summary.json", summary)
    progress.log("complete", **summary)
    print(json.dumps({"kept_after_all_filters": kept_count, "removal_reason_counts": reason_counts}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
