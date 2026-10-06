#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_ROOT = Path(os.environ.get("FEDRAG_CANONICAL_ROOT", "<REPO_ROOT>")).resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from math_qa import JinaAIClient
from scripts.run_gsm8k_small_router_sweep import parse_seed_list
from scripts.run_stabletoolbench_federated import (
    DEFAULT_STB_ROOT,
    DEFAULT_TOOLBENCH_INSTRUCTION_DIR,
    GROUPS,
    ProgressLogger,
    apply_junk_filter,
    assert_clean_tree,
    assert_no_eval_overlap,
    batched_query_embeddings,
    build_client_package,
    build_doc_package,
    build_flat_pool_package,
    build_ranked_candidates,
    build_tool_registry,
    candidate_selection_diagnostics,
    combine_packages,
    filter_eval_queries,
    filter_experience_items,
    limit_client_items,
    load_stabletoolbench_queries,
    load_toolbench_training_items,
    package_to_candidates,
    remove_exact_eval_overlaps,
    remove_near_duplicate_eval_overlaps,
    resolve_local_embedder,
    save_json,
    stable_hash,
    temporary_env,
)
from scripts.run_stabletoolbench_heldout import (
    HELDOUT_GROUPS,
    assign_clients,
    filter_heldout_pool,
    heldout_tool_set,
    item_mentions_heldout_tool,
)
from synapse.edge.aggregator import EdgeAggregator, EdgeConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="D4 split-sensitivity retrieval-only check for StableToolBench held-out split.")
    parser.add_argument("--stb-root", type=Path, default=DEFAULT_STB_ROOT)
    parser.add_argument("--toolbench-instruction-dir", type=Path, default=DEFAULT_TOOLBENCH_INSTRUCTION_DIR)
    parser.add_argument("--groups", type=str, default=",".join(GROUPS))
    parser.add_argument("--seeds", type=str, default="42,123,456")
    parser.add_argument("--filter-modes", type=str, default="strict,label_only,mention_only")
    parser.add_argument("--client-count", type=int, default=5)
    parser.add_argument("--max-items-per-client", type=int, default=5000)
    parser.add_argument("--partition-mode", choices=["category", "iid"], default="category")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retrieval-pool-size", type=int, default=200)
    parser.add_argument("--retrieval-mode", choices=["distinct_tool_topk"], default="distinct_tool_topk")
    parser.add_argument("--embed-model", type=str, default="jina-embeddings-v2-base-en")
    parser.add_argument("--pool-embedding-cache-dir", type=Path, default=CANONICAL_ROOT / "artifacts" / "cache" / "stabletoolbench")
    parser.add_argument("--contamination-near-duplicate-threshold", type=float, default=0.95)
    parser.add_argument("--contamination-report-threshold", type=float, default=0.90)
    parser.add_argument("--output-dir", type=Path, default=CANONICAL_ROOT / "artifacts" / "results" / "stabletoolbench_split_sensitivity_d4_r1")
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args()


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def filter_pool_by_mode(items: list[Any], heldout_tools: set[str], mode: str) -> tuple[list[Any], dict[str, Any]]:
    if mode == "strict":
        kept, info = filter_heldout_pool(items, heldout_tools)
        info["filter_mode"] = mode
        return kept, info
    kept = []
    dropped_label = []
    dropped_mention = []
    for item in items:
        label_hit = any(tool in heldout_tools for tool in item.gold_tools)
        mention_hit = item_mentions_heldout_tool(item, heldout_tools)
        if mode == "label_only" and label_hit:
            dropped_label.append(item)
            continue
        if mode == "mention_only" and mention_hit and not label_hit:
            dropped_mention.append(item)
            continue
        kept.append(item)
    remaining_label_hits = sum(1 for item in kept if any(tool in heldout_tools for tool in item.gold_tools))
    return kept, {
        "filter_mode": mode,
        "pool_items_before": len(items),
        "pool_items_dropped_heldout": len(dropped_label),
        "experience_entries_dropped_heldout": len(dropped_label) + len(dropped_mention),
        "experience_entries_dropped_heldout_mentions": len(dropped_mention),
        "pool_items_after": len(kept),
        "remaining_items_with_heldout_label": remaining_label_hits,
        "eval_queries_removed": 0,
    }


def retrieval_rows(name: str, package, queries, heldout_tools: set[str], jina_client: JinaAIClient, embed_model: str, args: argparse.Namespace, progress: ProgressLogger, *, seed: int, filter_mode: str) -> dict[str, Any]:
    started = time.perf_counter()
    progress.log("arm_prepare_begin", seed=seed, filter_mode=filter_mode, arm=name, artifact_count=len(package.artifacts))
    candidates = package_to_candidates(package, jina_client, embed_model)
    candidate_matrix = np.asarray([candidate.embedding for candidate in candidates], dtype=np.float32) if candidates else np.zeros((0, 0), dtype=np.float32)
    if candidate_matrix.size:
        norms = np.linalg.norm(candidate_matrix, axis=1)
        norms[norms == 0.0] = 1.0
        candidate_matrix = candidate_matrix / norms[:, None]
    query_embeddings = batched_query_embeddings(jina_client, [item.query for item in queries], embed_model)
    progress.log("arm_prepare_done", seed=seed, filter_mode=filter_mode, arm=name, elapsed_s=time.perf_counter() - started)
    rows = []
    for idx, (item, embedding) in enumerate(zip(queries, query_embeddings), start=1):
        q = np.asarray(embedding, dtype=np.float32)
        qn = float(np.linalg.norm(q))
        if qn > 0.0:
            q = q / qn
        similarities = candidate_matrix @ q if candidate_matrix.size else np.asarray([], dtype=np.float32)
        ranked, pool_tools = build_ranked_candidates(candidates, similarities, args.retrieval_pool_size, args.top_k, args.retrieval_mode)
        diag = candidate_selection_diagnostics(candidates, similarities, ranked, args.top_k, args.retrieval_pool_size, args.retrieval_mode)
        if diag["candidate_shortfall"]:
            raise RuntimeError(f"{filter_mode}/{name} seed {seed} query {item.query_id} returned {diag['candidate_distinct_count']} distinct candidates")
        tools = [candidate.parent_tool for candidate in ranked]
        gold = item.gold_tools
        subset = "heldout" if gold and all(tool in heldout_tools for tool in gold) else "labeled"
        rows.append({
            "query_id": item.query_id,
            "query_text": item.query,
            "group": item.group,
            "gold_tools": gold,
            "gold_parent_tools": gold,
            "subset": subset,
            "candidate_tools": tools,
            "candidate_ids": [candidate.candidate_id for candidate in ranked],
            "retrieval_pool_tools": pool_tools,
            "gold_in_top_k": any(tool in gold for tool in tools),
            "retrieval_top1_correct": bool(tools and tools[0] in gold),
            **diag,
        })
        if idx == 1 or idx % 100 == 0 or idx == len(queries):
            progress.log("arm_progress", seed=seed, filter_mode=filter_mode, arm=name, completed_queries=idx, total_queries=len(queries), running_recall_at_5=sum(row["gold_in_top_k"] for row in rows) / len(rows))
    result = {"arm": name, "seed": seed, "filter_mode": filter_mode, "rows": rows}
    result["metrics"] = summarize(rows)
    return result


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for subset in ("heldout", "labeled", "all"):
        bucket = rows if subset == "all" else [row for row in rows if row["subset"] == subset]
        out[subset] = {
            "n": len(bucket),
            "recall_at_5": sum(row["gold_in_top_k"] for row in bucket) / len(bucket) if bucket else 0.0,
            "retrieval_top1": sum(row["retrieval_top1_correct"] for row in bucket) / len(bucket) if bucket else 0.0,
            "candidate_distinct_mean": statistics.mean([row["candidate_distinct_count"] for row in bucket]) if bucket else 0.0,
            "candidate_lt5_share": sum(row["candidate_distinct_count"] < 5 for row in bucket) / len(bucket) if bucket else 0.0,
            "candidate_depth_mean": statistics.mean([row["candidate_depth_reached"] for row in bucket]) if bucket else 0.0,
        }
    return out


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for mode in sorted({result["filter_mode"] for result in results}):
        out[mode] = {}
        for arm in sorted({result["arm"] for result in results if result["filter_mode"] == mode}):
            arm_results = [result for result in results if result["filter_mode"] == mode and result["arm"] == arm]
            out[mode][arm] = {}
            for subset in ("heldout", "labeled", "all"):
                vals = [result["metrics"][subset]["recall_at_5"] for result in arm_results]
                top1 = [result["metrics"][subset]["retrieval_top1"] for result in arm_results]
                out[mode][arm][subset] = {
                    "mean_recall_at_5": statistics.mean(vals) if vals else 0.0,
                    "sd_recall_at_5": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                    "mean_retrieval_top1": statistics.mean(top1) if top1 else 0.0,
                    "sd_retrieval_top1": statistics.stdev(top1) if len(top1) > 1 else 0.0,
                    "n_per_seed": [result["metrics"][subset]["n"] for result in arm_results],
                }
    return out


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    commit, dirty = assert_clean_tree(allow_dirty=args.allow_dirty)
    progress = ProgressLogger(args.output_dir)
    seeds = parse_seed_list(args.seeds)
    groups = parse_csv(args.groups)
    filter_modes = parse_csv(args.filter_modes)
    progress.log("launch", repo_commit=commit, dirty_entry_count=len(dirty), seeds=seeds, filter_modes=filter_modes)
    embedder_info = resolve_local_embedder()
    jina_client = JinaAIClient(api_keys=[])
    with temporary_env({"JINA_LOCAL_EMBED_MODEL": embedder_info["model_path"], "JINA_LOCAL_EMBED_DEVICE": embedder_info["device"], "JINA_LOCAL_EMBED_LOCAL_ONLY": embedder_info["local_only"], "JINA_API_KEY": None}):
        queries_raw = load_stabletoolbench_queries(args.stb_root, groups)
        train_items_raw = load_toolbench_training_items(args.toolbench_instruction_dir)
        registry = build_tool_registry(queries_raw + train_items_raw)
        registry, junk_info = apply_junk_filter(registry)
        queries, eval_filter = filter_eval_queries(queries_raw, registry)
        train_items, train_filter = filter_experience_items(train_items_raw, registry)
        heldout_tools, heldout_sanity = heldout_tool_set(queries, HELDOUT_GROUPS)
        heldout_set = set(heldout_tools)
        heldout_sha = stable_hash(heldout_tools)
        progress.log("data_ready", test_count=len(queries), train_count=len(train_items), heldout_tool_count=len(heldout_tools), heldout_tools_sha256=heldout_sha, junk_filter=junk_info, eval_filter=eval_filter, train_filter=train_filter)
        train_items, exact_info = remove_exact_eval_overlaps(train_items, queries)
        train_items, near_info, forbidden = remove_near_duplicate_eval_overlaps(
            train_items,
            queries,
            jina_client,
            args.embed_model,
            removal_threshold=args.contamination_near_duplicate_threshold,
            report_threshold=args.contamination_report_threshold,
            pool_embedding_cache_dir=args.pool_embedding_cache_dir,
            progress=progress,
        )
        leak_filter = {**exact_info, **near_info}
        leak_filter["pool_items_removed_total"] = int(leak_filter["pool_items_removed_exact"]) + int(leak_filter["pool_items_removed_near_dup"])
        leak_filter["eval_queries_removed"] = 0
        leak_filter["post_filter_refusal_check"] = assert_no_eval_overlap(train_items, queries, forbidden_near_duplicate_texts=forbidden, stage="d4_post_leak_filter_pool")
        progress.log("contamination_filter_done", train_count=len(train_items), test_count=len(queries), **leak_filter)
        doc_package, doc_hash = build_doc_package(registry)
        flat_package, flat_hash = build_flat_pool_package(registry)
        all_results: list[dict[str, Any]] = []
        for mode in filter_modes:
            mode_items, heldout_filter = filter_pool_by_mode(train_items, heldout_set, mode)
            heldout_filter["heldout_tools_sha256"] = heldout_sha
            heldout_filter["pool_sha256"] = stable_hash([{"query_id": item.query_id, "query": item.query, "gold_tools": item.gold_tools} for item in mode_items])
            progress.log("heldout_pool_filter_done", **heldout_filter)
            for seed in seeds:
                clients = limit_client_items(assign_clients(mode_items, args.client_count, args.partition_mode, seed), args.max_items_per_client, seed)
                client_packages = []
                client_hashes = []
                for client_id, items in sorted(clients.items()):
                    package, package_hash = build_client_package(client_id, items, registry, jina_client, args.embed_model)
                    client_packages.append(package)
                    client_hashes.append(package_hash)
                with temporary_env({"SYNAPSE_EDGE_MERGE_POLICY": "conflict_log"}):
                    merged = EdgeAggregator(EdgeConfig(edge_id=f"d4_{mode}_{seed}")).merge_packages(client_packages)
                shared_package, shared_hash = combine_packages("synapse_with_docs", [doc_package, merged])
                for arm, package, package_hash in (("docs_only", flat_package, flat_hash), ("shared", shared_package, shared_hash)):
                    result = retrieval_rows(arm, package, queries, heldout_set, jina_client, args.embed_model, args, progress, seed=seed, filter_mode=mode)
                    result.update({
                        "paper_eligible": True,
                        "repo_commit": commit,
                        "data_mode": "toolbench_train",
                        "heldout_tools_sha256": heldout_sha,
                        "heldout_sanity": heldout_sanity,
                        "heldout_filter": heldout_filter,
                        "contamination_filter": leak_filter,
                        "candidate_rule": "distinct5_walkdown",
                        "retriever": "jina",
                        "compendium": {"global_sha256": package_hash, "tool_doc_sha256": doc_hash, "client_sha256": client_hashes},
                    })
                    save_json(args.output_dir / mode / f"seed_{seed}" / f"{arm}.json", result)
                    all_results.append(result)
                    progress.log("arm_done", filter_mode=mode, seed=seed, arm=arm, heldout_recall_at_5=result["metrics"]["heldout"]["recall_at_5"], labeled_recall_at_5=result["metrics"]["labeled"]["recall_at_5"])
        summary = {
            "paper_eligible": True,
            "config": {
                "repo_commit": commit,
                "seeds": seeds,
                "filter_modes": filter_modes,
                "retriever": "jina",
                "candidate_rule": "distinct5_walkdown",
                "retrieval_pool_size": args.retrieval_pool_size,
                "top_k": args.top_k,
                "heldout_tools_sha256": heldout_sha,
                "local_embedder": embedder_info,
            },
            "aggregate": aggregate(all_results),
        }
        save_json(args.output_dir / "summary.json", summary)
        progress.log("complete", output_dir=str(args.output_dir))
        print(json.dumps(summary["aggregate"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
