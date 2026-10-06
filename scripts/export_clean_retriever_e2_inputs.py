#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_ROOT = Path(os.environ.get("FEDRAG_CANONICAL_ROOT", REPO_ROOT)).resolve()
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
    remove_exact_eval_overlaps,
    remove_near_duplicate_eval_overlaps,
    resolve_local_embedder,
    stable_hash,
    temporary_env,
    tool_description,
)
from scripts.run_stabletoolbench_heldout import HELDOUT_GROUPS, filter_heldout_pool, heldout_tool_set


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export clean held-out-excluded inputs for E2 BGE retriever training.")
    p.add_argument("--stb-root", type=Path, default=DEFAULT_STB_ROOT)
    p.add_argument("--toolbench-instruction-dir", type=Path, default=DEFAULT_TOOLBENCH_INSTRUCTION_DIR)
    p.add_argument("--groups", default=",".join(GROUPS))
    p.add_argument("--output-dir", type=Path, default=CANONICAL_ROOT / "artifacts" / "results" / "clean_retriever_e2_inputs_r1")
    p.add_argument("--embed-model", default="jina-embeddings-v2-base-en")
    p.add_argument("--pool-embedding-cache-dir", type=Path, default=CANONICAL_ROOT / "artifacts" / "cache" / "stabletoolbench")
    p.add_argument("--contamination-near-duplicate-threshold", type=float, default=0.95)
    p.add_argument("--contamination-report-threshold", type=float, default=0.90)
    p.add_argument("--allow-dirty", action="store_true")
    return p.parse_args()


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


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
    progress.log("loaded", raw_query_count=len(queries), raw_train_count=len(train_items), raw_registry_tool_count=len(registry))

    registry, registry_info = apply_junk_filter(registry)
    queries, query_info = filter_eval_queries(queries, registry)
    train_items, train_info = filter_experience_items(train_items, registry)
    progress.log("junk_filter_done", query_count=len(queries), train_count=len(train_items), registry_tool_count=len(registry), **registry_info, **query_info, **train_info)

    embedder_info = resolve_local_embedder()
    jina = JinaAIClient(api_keys=[])
    local_env = {
        "JINA_LOCAL_EMBED_MODEL": embedder_info["model_path"],
        "JINA_LOCAL_EMBED_DEVICE": embedder_info["device"],
        "JINA_LOCAL_EMBED_LOCAL_ONLY": embedder_info["local_only"],
        "JINA_API_KEY": None,
    }
    with temporary_env(local_env):
        train_items, exact_info = remove_exact_eval_overlaps(train_items, queries)
        train_items, near_info, forbidden = remove_near_duplicate_eval_overlaps(
            train_items,
            queries,
            jina,
            args.embed_model,
            removal_threshold=args.contamination_near_duplicate_threshold,
            report_threshold=args.contamination_report_threshold,
            pool_embedding_cache_dir=args.pool_embedding_cache_dir,
            progress=progress,
        )
    near_info.pop("pool_items_removed_near_dup_query_ids", None)
    progress.log("contamination_filter_done", train_count=len(train_items), **exact_info, **near_info)

    heldout_tools, heldout_sanity = heldout_tool_set(queries, HELDOUT_GROUPS)
    heldout_set = set(heldout_tools)
    filtered_items, heldout_filter = filter_heldout_pool(train_items, heldout_set)
    progress.log("heldout_filter_done", train_count=len(filtered_items), heldout_tool_count=len(heldout_tools), **heldout_filter)

    pairs_path = args.output_dir / "pairs.jsonl"
    with pairs_path.open("w", encoding="utf-8") as handle:
        for item in filtered_items:
            handle.write(json.dumps({"item_id": item.query_id, "query": item.query, "gold_tools": item.gold_tools}, sort_keys=True) + "\n")

    tool_docs = {tool: tool_description(doc) for tool, doc in sorted(registry.items())}
    tool_docs_path = args.output_dir / "tool_docs.json"
    heldout_path = args.output_dir / "heldout_tools.json"
    tests_path = args.output_dir / "test_queries.json"
    write_json(tool_docs_path, tool_docs)
    write_json(heldout_path, heldout_tools)
    write_json(tests_path, [query.query for query in queries])

    manifest = {
        "repo_commit": commit,
        "dirty_entry_count": len(dirty),
        "paper_eligible": True,
        "groups": groups,
        "query_count": len(queries),
        "pair_count": len(filtered_items),
        "tool_doc_count": len(tool_docs),
        "heldout_tool_count": len(heldout_tools),
        "heldout_tools_sha256": stable_hash(heldout_tools),
        "heldout_sanity": heldout_sanity,
        "junk_filter": {**registry_info, **query_info, **train_info},
        "contamination_filter": {**exact_info, **near_info, "forbidden_near_duplicate_text_count": len(forbidden)},
        "heldout_filter": heldout_filter,
        "paths": {
            "pairs": str(pairs_path),
            "tool_docs": str(tool_docs_path),
            "heldout": str(heldout_path),
            "test_queries": str(tests_path),
        },
        "sha256": {
            "pairs": sha256_file(pairs_path),
            "tool_docs": sha256_file(tool_docs_path),
            "heldout": sha256_file(heldout_path),
            "test_queries": sha256_file(tests_path),
        },
    }
    write_json(args.output_dir / "manifest.json", manifest)
    progress.log("complete", pair_count=len(filtered_items), tool_doc_count=len(tool_docs), heldout_tool_count=len(heldout_tools), paths=manifest["paths"], sha256=manifest["sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
