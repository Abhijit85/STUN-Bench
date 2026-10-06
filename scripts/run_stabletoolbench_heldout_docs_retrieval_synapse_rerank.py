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
from scripts.run_reranker_prompt_sweep import run_prompt
from scripts.run_stabletoolbench_federated import (
    DEFAULT_MODEL_PATH,
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
    combine_packages,
    filter_eval_queries,
    filter_experience_items,
    hash_package,
    limit_client_items,
    load_local_backend,
    load_stabletoolbench_queries,
    load_toolbench_training_items,
    maybe_cuda_synchronize,
    package_to_candidates,
    remove_exact_eval_overlaps,
    remove_near_duplicate_eval_overlaps,
    resolve_local_embedder,
    save_json,
    stable_hash,
    summarize_rows,
    temporary_env,
)
from scripts.run_stabletoolbench_heldout import (
    HELDOUT_GROUPS,
    add_subset_fields,
    assign_clients,
    filter_heldout_pool,
    heldout_tool_set,
)
from synapse.edge.aggregator import EdgeAggregator, EdgeConfig

DEFAULT_OUTPUT_DIR = CANONICAL_ROOT / "artifacts" / "results" / "stabletoolbench_heldout_docs_retrieval_synapse_rerank_r1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E5b: Docs-only retrieval with Synapse typed reranking on held-out tools.")
    parser.add_argument("--stb-root", type=Path, default=DEFAULT_STB_ROOT)
    parser.add_argument("--toolbench-instruction-dir", type=Path, default=DEFAULT_TOOLBENCH_INSTRUCTION_DIR)
    parser.add_argument("--groups", type=str, default=",".join(GROUPS))
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--client-count", type=int, default=5)
    parser.add_argument("--max-items-per-client", type=int, default=5000)
    parser.add_argument("--partition-mode", type=str, default="category", choices=["category", "iid"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retrieval-pool-size", type=int, default=20)
    parser.add_argument("--retrieval-mode", type=str, default="distinct_tool_topk")
    parser.add_argument("--reranker-variant", type=str, default="V3")
    parser.add_argument("--merge-policy", type=str, default="conflict_log", choices=["conflict_log", "round_delayed", "majority"])
    parser.add_argument("--embed-model", type=str, default="jina-embeddings-v2-base-en")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--pool-embedding-cache-dir", type=Path, default=CANONICAL_ROOT / "artifacts" / "cache" / "stabletoolbench")
    parser.add_argument("--contamination-near-duplicate-threshold", type=float, default=0.95)
    parser.add_argument("--contamination-report-threshold", type=float, default=0.90)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args()


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_int_csv(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def normalize_matrix(vectors: list[list[float]]) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.size:
        norms = np.linalg.norm(matrix, axis=1)
        norms[norms == 0.0] = 1.0
        matrix = matrix / norms[:, None]
    return matrix


def docs_ranked_candidates(package, queries, jina_client, embed_model: str, retrieval_pool_size: int, top_k: int, retrieval_mode: str):
    candidates = package_to_candidates(package, jina_client, embed_model)
    candidate_matrix = normalize_matrix([candidate.embedding for candidate in candidates]) if candidates else np.zeros((0, 0), dtype=np.float32)
    query_embeddings = batched_query_embeddings(jina_client, [item.query for item in queries], embed_model) if queries else []
    rows = []
    for item, embedding in zip(queries, query_embeddings):
        q = np.asarray(embedding, dtype=np.float32)
        qn = float(np.linalg.norm(q))
        if qn > 0.0:
            q = q / qn
        similarities = candidate_matrix @ q if candidate_matrix.size else np.asarray([], dtype=np.float32)
        ranked, pool_tools = build_ranked_candidates(candidates, similarities, retrieval_pool_size, top_k, retrieval_mode)
        rows.append(
            {
                "query_id": item.query_id,
                "query_text": item.query,
                "group": item.group,
                "gold_parent_tools": item.gold_tools,
                "candidate_tools": [candidate.parent_tool for candidate in ranked],
                "candidate_ids": [candidate.candidate_id for candidate in ranked],
                "retrieval_pool_tools": pool_tools,
                "docs_gold_in_top_k": any(tool in item.gold_tools for tool in [candidate.parent_tool for candidate in ranked]),
            }
        )
    return rows


def first_candidate_by_tool(package, jina_client, embed_model: str):
    mapping = {}
    for candidate in package_to_candidates(package, jina_client, embed_model):
        mapping.setdefault(candidate.parent_tool, candidate)
    return mapping


def rerank_docs_candidates_with_synapse(*, docs_rows, synapse_by_tool, backend, args, progress: ProgressLogger, seed: int, heldout_tools: set[str]):
    rows = []
    parse_failures = 0
    missing_tools = 0
    total_latency = 0.0
    for idx, row in enumerate(docs_rows, start=1):
        started = time.perf_counter()
        ranked = []
        missing = []
        for tool in row["candidate_tools"][: args.top_k]:
            candidate = synapse_by_tool.get(tool)
            if candidate is None:
                missing.append(tool)
                continue
            ranked.append(candidate)
        missing_tools += len(missing)
        top_candidate = ranked[0] if ranked else None
        if top_candidate is None:
            predicted_tool = ""
            prompt_hash = ""
            parse_ok = False
            fallback = True
            rerank_s = 0.0
        else:
            maybe_cuda_synchronize(backend)
            rerank_started = time.perf_counter()
            result = run_prompt(backend, args.reranker_variant, "toolbench", row["query_text"], ranked, [], top_candidate)
            maybe_cuda_synchronize(backend)
            rerank_s = time.perf_counter() - rerank_started
            predicted_tool = result.predicted_tool
            prompt_hash = result.prompt_hash
            parse_ok = result.parse_ok
            fallback = result.fallback_used
            parse_failures += int(not parse_ok)
        gold = row["gold_parent_tools"]
        subset = "heldout" if any(tool in heldout_tools for tool in gold) else "labeled"
        rows.append(
            {
                "query_id": row["query_id"],
                "query_text": row["query_text"],
                "group": row["group"],
                "gold_parent_tools": gold,
                "predicted_tool": predicted_tool,
                "routed_correctly": predicted_tool in gold,
                "correct": predicted_tool in gold,
                "gold_in_top_k": row["docs_gold_in_top_k"],
                "docs_retrieval_candidate_tools": row["candidate_tools"],
                "docs_retrieval_candidate_ids": row["candidate_ids"],
                "top_candidates": [candidate.parent_tool for candidate in ranked],
                "top_candidate_ids": [candidate.candidate_id for candidate in ranked],
                "retrieval_pool_tools": row["retrieval_pool_tools"],
                "missing_synapse_tools": missing,
                "subset": subset,
                "parse_ok": parse_ok,
                "fallback_used": fallback,
                "retrieval_s": 0.0,
                "rerank_s": rerank_s,
                "total_s": time.perf_counter() - started,
                "prompt_hash": prompt_hash,
            }
        )
        total_latency += rows[-1]["total_s"]
        if idx == 1 or idx % 25 == 0 or idx == len(docs_rows):
            progress.log(
                "rerank_progress",
                seed=seed,
                completed_queries=idx,
                total_queries=len(docs_rows),
                running_accuracy=sum(1 for item in rows if item["routed_correctly"]) / len(rows),
                running_recall_at_5=sum(1 for item in rows if item["gold_in_top_k"]) / len(rows),
            )
    summary = summarize_rows(rows)
    summary.update(
        {
            "arm": "docs_retrieval_synapse_rerank",
            "seed": seed,
            "rows": rows,
            "subset_metrics": {
                subset: {
                    "n": len(subset_rows),
                    "accuracy": sum(1 for item in subset_rows if item["routed_correctly"]) / len(subset_rows) if subset_rows else 0.0,
                    "recall_at_5": sum(1 for item in subset_rows if item["gold_in_top_k"]) / len(subset_rows) if subset_rows else 0.0,
                }
                for subset, subset_rows in (
                    ("heldout", [item for item in rows if item["subset"] == "heldout"]),
                    ("labeled", [item for item in rows if item["subset"] == "labeled"]),
                )
            },
            "parse_failure_rate": parse_failures / len(rows) if rows else 0.0,
            "missing_synapse_tool_mentions": missing_tools,
            "mean_latency_seconds": total_latency / len(rows) if rows else 0.0,
            "retrieval_backend": "flat_pool_docs_only",
            "rerank_backend": "synapse_typed_fields",
        }
    )
    return summary


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for subset in ("heldout", "labeled"):
        accs = [result["subset_metrics"][subset]["accuracy"] for result in results if subset in result.get("subset_metrics", {})]
        recs = [result["subset_metrics"][subset]["recall_at_5"] for result in results if subset in result.get("subset_metrics", {})]
        ns = [result["subset_metrics"][subset]["n"] for result in results if subset in result.get("subset_metrics", {})]
        out[subset] = {
            "mean_n": statistics.mean(ns) if ns else 0.0,
            "mean_accuracy": statistics.mean(accs) if accs else 0.0,
            "sd_accuracy": statistics.stdev(accs) if len(accs) > 1 else 0.0,
            "mean_recall_at_5": statistics.mean(recs) if recs else 0.0,
            "sd_recall_at_5": statistics.stdev(recs) if len(recs) > 1 else 0.0,
        }
    return out


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    commit, dirty = assert_clean_tree(allow_dirty=args.allow_dirty)
    progress = ProgressLogger(args.output_dir)
    progress.log("launch", git_commit=commit, dirty_entry_count=len(dirty))
    groups = parse_csv(args.groups)
    seeds = parse_int_csv(args.seeds)
    embedder_info = resolve_local_embedder()
    jina_client = JinaAIClient(api_keys=[])
    with temporary_env(
        {
            "JINA_LOCAL_EMBED_MODEL": embedder_info["model_path"],
            "JINA_LOCAL_EMBED_DEVICE": embedder_info["device"],
            "JINA_LOCAL_EMBED_LOCAL_ONLY": embedder_info["local_only"],
            "JINA_API_KEY": None,
        }
    ):
        all_queries = load_stabletoolbench_queries(args.stb_root, groups)
        registry = build_tool_registry(all_queries)
        train_items = load_toolbench_training_items(args.toolbench_instruction_dir)
        registry.update(build_tool_registry(train_items))
        registry, junk_info = apply_junk_filter(registry)
        queries, eval_filter = filter_eval_queries(all_queries, registry)
        train_items, train_filter = filter_experience_items(train_items, registry)
        progress.log("junk_filter_done", test_count=len(queries), train_count=len(train_items), registry_size=len(registry), junk_filter=junk_info, eval_filter=eval_filter, train_filter=train_filter)
        heldout_tools, heldout_sanity = heldout_tool_set(queries, HELDOUT_GROUPS)
        heldout_set = set(heldout_tools)
        heldout_sha = stable_hash(heldout_tools)
        exact_forbidden = {item.query for item in queries}
        train_items, exact_info = remove_exact_eval_overlaps(train_items, queries)
        train_items, near_info, forbidden_near_duplicate_texts = remove_near_duplicate_eval_overlaps(
            train_items,
            queries,
            jina_client,
            args.embed_model,
            removal_threshold=args.contamination_near_duplicate_threshold,
            report_threshold=args.contamination_report_threshold,
            pool_embedding_cache_dir=args.pool_embedding_cache_dir,
            progress=progress,
        )
        near_info.pop("pool_items_removed_near_dup_query_ids", None)
        leak_filter = {**exact_info, **near_info}
        leak_filter["pool_items_removed_total"] = int(leak_filter["pool_items_removed_exact"]) + int(leak_filter["pool_items_removed_near_dup"])
        leak_filter["eval_queries_removed"] = 0
        leak_filter["post_filter_refusal_check"] = assert_no_eval_overlap(train_items, queries, forbidden_near_duplicate_texts=forbidden_near_duplicate_texts, stage="e5b_post_leak_filter_pool")
        progress.log("contamination_filter_done", train_count=len(train_items), test_count=len(queries), **leak_filter)
        train_items, heldout_filter = filter_heldout_pool(train_items, heldout_set)
        if heldout_filter["remaining_items_with_heldout_label"] != 0:
            raise RuntimeError("heldout label filter failed")
        heldout_filter["eval_queries_removed"] = 0
        heldout_filter["heldout_tools_sha256"] = heldout_sha
        heldout_filter["pool_sha256"] = stable_hash([{"query_id": item.query_id, "query": item.query, "gold_tools": item.gold_tools} for item in train_items])
        progress.log("heldout_pool_filter_done", **heldout_filter)
        progress.log("load_backend_begin", model_path=args.model_path)
        backend = load_local_backend(args.model_path)
        progress.log("load_backend_done")
        tool_doc_package, tool_doc_hash = build_doc_package(registry)
        flat_package, flat_hash = build_flat_pool_package(registry)
        results = []
        for seed in seeds:
            progress.log("seed_begin", seed=seed)
            clients = assign_clients(train_items, args.client_count, args.partition_mode, seed)
            clients = limit_client_items(clients, args.max_items_per_client, seed)
            client_packages = []
            client_hashes = []
            for client_id, items in sorted(clients.items()):
                progress.log("client_package_begin", seed=seed, client_id=client_id, item_count=len(items))
                package, package_hash = build_client_package(client_id, items, registry, jina_client, args.embed_model)
                client_packages.append(package)
                client_hashes.append(package_hash)
                progress.log("client_package_done", seed=seed, client_id=client_id, artifact_count=len(package.artifacts), package_sha256=package_hash)
            with temporary_env({"SYNAPSE_EDGE_MERGE_POLICY": args.merge_policy}):
                progress.log("arm_merge_begin", seed=seed, arm="synapse", merge_policy=args.merge_policy)
                merged = EdgeAggregator(EdgeConfig(edge_id=f"stabletoolbench_e5b_seed_{seed}")).merge_packages(client_packages)
            if merged is None:
                raise RuntimeError(f"seed {seed} synapse merge produced no package")
            synapse_package, synapse_hash = combine_packages("synapse_with_docs", [tool_doc_package, merged])
            heldout_queries = [query for query in queries if any(tool in heldout_set for tool in query.gold_tools)]
            docs_rows = docs_ranked_candidates(flat_package, heldout_queries, jina_client, args.embed_model, args.retrieval_pool_size, args.top_k, args.retrieval_mode)
            docs_candidates_hash = stable_hash(docs_rows)
            progress.log("docs_candidates_ready", seed=seed, query_count=len(docs_rows), docs_candidates_sha256=docs_candidates_hash)
            synapse_by_tool = first_candidate_by_tool(synapse_package, jina_client, args.embed_model)
            result = rerank_docs_candidates_with_synapse(docs_rows=docs_rows, synapse_by_tool=synapse_by_tool, backend=backend, args=args, progress=progress, seed=seed, heldout_tools=heldout_set)
            result.update(
                {
                    "paper_eligible": True,
                    "repo_commit": commit,
                    "data_mode": "toolbench_train",
                    "heldout_tools_sha256": heldout_sha,
                    "heldout_tool_count": len(heldout_tools),
                    "heldout_sanity": heldout_sanity,
                    "heldout_filter": heldout_filter,
                    "contamination_filter": leak_filter,
                    "eval_queries_removed": 0,
                    "embedder": embedder_info,
                    "retrieval_mode": args.retrieval_mode,
                    "retrieval_pool_size": args.retrieval_pool_size,
                    "top_k": args.top_k,
                    "reranker_variant": args.reranker_variant,
                    "source_packages": {
                        "flat_pool_sha256": flat_hash,
                        "tool_doc_sha256": tool_doc_hash,
                        "synapse_sha256": synapse_hash,
                        "client_sha256": client_hashes,
                        "docs_candidates_sha256": docs_candidates_hash,
                    },
                    "metric_definition": {"correct": "predicted_tool in gold_parent_tools", "heldout": "all gold tools in H"},
                }
            )
            add_subset_fields(result, heldout_set)
            out_path = args.output_dir / f"seed_{seed}" / "docs_retrieval_synapse_rerank.json"
            save_json(out_path, result)
            results.append(result)
            progress.log("seed_complete", seed=seed, accuracy=result["accuracy"], heldout_accuracy=result["subset_metrics"]["heldout"]["accuracy"], heldout_recall_at_5=result["subset_metrics"]["heldout"]["recall_at_5"])
        summary = {
            "paper_eligible": True,
            "config": {
                "repo_commit": commit,
                "seeds": seeds,
                "groups": groups,
                "heldout_groups": HELDOUT_GROUPS,
                "heldout_tools_sha256": heldout_sha,
                "top_k": args.top_k,
                "retrieval_pool_size": args.retrieval_pool_size,
                "retrieval_mode": args.retrieval_mode,
                "reranker_variant": args.reranker_variant,
                "merge_policy": args.merge_policy,
                "local_embedder": embedder_info,
            },
            "aggregate": aggregate(results),
        }
        save_json(args.output_dir / "summary.json", summary)
        progress.log("complete", output_dir=str(args.output_dir))
        print(json.dumps(summary["aggregate"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
