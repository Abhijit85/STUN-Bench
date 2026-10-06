#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_ROOT = Path(os.environ.get('FEDRAG_CANONICAL_ROOT', '<REPO_ROOT>')).resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from math_qa import JinaAIClient
from scripts.run_reranker_prompt_sweep import run_prompt
from scripts.run_stabletoolbench_federated import (
    DEFAULT_MODEL_PATH,
    DEFAULT_STB_ROOT,
    ProgressLogger,
    assert_clean_tree,
    build_ranked_candidates,
    merge_heartbeat,
    maybe_cuda_synchronize,
    package_to_candidates,
    summarize_rows,
    load_local_backend,
    temporary_env,
    resolve_local_embedder,
    save_json,
    combine_packages,
    load_package_file,
    save_package,
)
from scripts.run_stabletoolbench_typing_isolation import load_eval_queries, load_seed_packages
from synapse.edge.aggregator import EdgeAggregator, EdgeConfig

DEFAULT_OUTPUT_DIR = CANONICAL_ROOT / "artifacts" / "results" / "stabletoolbench_oracle_r1"
PACKAGE_ROOTS = {
    42: CANONICAL_ROOT / "artifacts" / "verification" / "stabletoolbench_clean_anchor_seed42_r4" / "packages",
    123: CANONICAL_ROOT / "artifacts" / "verification" / "stabletoolbench_clean_anchor_seed123_r4" / "packages",
    456: CANONICAL_ROOT / "artifacts" / "verification" / "stabletoolbench_clean_anchor_seed456_r4" / "packages",
}
A1_R16_ROOTS = {
    42: Path("<REPO_ROOT>_runs/artifacts/verification/stabletoolbench_a1_seed42_eval_r16/seed_42"),
    123: Path("<REPO_ROOT>_runs/artifacts/verification/stabletoolbench_a1_seed123_eval_r16/seed_123"),
    456: Path("<REPO_ROOT>_runs/artifacts/verification/stabletoolbench_a1_seed456_eval_r16/seed_456"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="StableToolBench oracle-retrieval replay over clean A1 compendiums.")
    parser.add_argument("--seeds", type=str, default="42,123,456")
    parser.add_argument("--arms", type=str, default="synapse,centralized,flat_pool")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retrieval-pool-size", type=int, default=5)
    parser.add_argument("--retrieval-mode", type=str, default="distinct_tool_topk")
    parser.add_argument("--reranker-variant", type=str, default="V3")
    parser.add_argument("--embed-model", type=str, default="jina-embeddings-v2-base-en")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--stb-root", type=Path, default=DEFAULT_STB_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args()


def parse_csv_int(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def load_json(path: Path) -> Any:
    import json
    return json.loads(path.read_text(encoding="utf-8"))


def verify_hash(name: str, expected: str, actual: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{name} package hash mismatch: expected {expected}, got {actual}")


def build_seed_packages(seed: int, progress: ProgressLogger | None = None) -> tuple[dict[str, Any], Any, Any, list[Any]]:
    packages_root = PACKAGE_ROOTS[seed]
    seed_dir = packages_root / f"seed_{seed}"
    synapse_cache = seed_dir / "synapse_conflictlog_cached.json"
    centralized_cache = seed_dir / "centralized_cached.json"
    if progress is not None:
        progress.log("seed_package_load_begin", seed=seed, packages_root=str(packages_root))
    client_packages, tool_doc_package, tool_doc_hash = load_seed_packages(packages_root, seed)
    flat_package, flat_hash = load_package_file(seed_dir / "flat_pool.json")
    if progress is not None:
        progress.log("seed_package_load_done", seed=seed, client_count=len(client_packages), tool_doc_artifacts=len(tool_doc_package.artifacts), flat_artifacts=len(flat_package.artifacts))
    if centralized_cache.exists():
        centralized_package, centralized_hash = load_package_file(centralized_cache)
        pooled_hash = str((centralized_package.metadata or {}).get('pooled_experience_sha256') or '')
        if progress is not None:
            progress.log("centralized_cache_hit", seed=seed, path=str(centralized_cache), artifact_count=len(centralized_package.artifacts))
    else:
        if progress is not None:
            progress.log("centralized_build_begin", seed=seed)
        pooled_experience, pooled_hash = combine_packages("centralized_experience", list(client_packages.values()))
        centralized_package, centralized_hash = combine_packages("centralized_with_docs", [tool_doc_package, pooled_experience])
        centralized_package = type(centralized_package)(source_id=centralized_package.source_id, artifacts=centralized_package.artifacts, metadata={**dict(centralized_package.metadata or {}), 'pooled_experience_sha256': pooled_hash})
        save_package(centralized_cache, centralized_package, centralized_hash)
        if progress is not None:
            progress.log("centralized_build_done", seed=seed, path=str(centralized_cache), artifact_count=len(centralized_package.artifacts))
    if synapse_cache.exists():
        synapse_package, synapse_hash = load_package_file(synapse_cache)
        aggregator = None
        merged = None
        if progress is not None:
            progress.log("synapse_cache_hit", seed=seed, path=str(synapse_cache), artifact_count=len(synapse_package.artifacts))
    else:
        if progress is not None:
            progress.log("synapse_merge_begin", seed=seed, client_count=len(client_packages))
        with temporary_env({"SYNAPSE_EDGE_MERGE_POLICY": "conflict_log"}):
            aggregator = EdgeAggregator(EdgeConfig(edge_id=f"oracle_seed_{seed}"))
            if progress is not None:
                with merge_heartbeat(progress, seed=seed, arm="synapse", merge_policy="conflict_log"):
                    merged = aggregator.merge_packages(list(client_packages.values()))
            else:
                merged = aggregator.merge_packages(list(client_packages.values()))
        if merged is None:
            raise RuntimeError(f"Seed {seed} synapse merge produced no package")
        synapse_package, synapse_hash = combine_packages("synapse_with_docs", [tool_doc_package, merged])
        synapse_package = type(synapse_package)(source_id=synapse_package.source_id, artifacts=synapse_package.artifacts, metadata={**dict(synapse_package.metadata or {}), 'merge_policy': 'conflict_log'})
        save_package(synapse_cache, synapse_package, synapse_hash)
        if progress is not None:
            progress.log("synapse_merge_done", seed=seed, path=str(synapse_cache), artifact_count=len(synapse_package.artifacts), edge_conflict_count=len(getattr(aggregator, 'conflict_log', []) or []))
    return client_packages, tool_doc_package, {
        "flat_pool": (flat_package, flat_hash),
        "centralized": (centralized_package, centralized_hash),
        "synapse": (synapse_package, synapse_hash),
        "tool_doc_sha256": tool_doc_hash,
        "pooled_experience_sha256": pooled_hash,
    }, aggregator, [merged] if merged is not None else []


def evaluate_oracle_arm(name: str, package, test_items, jina_client, embed_model, backend, top_k, retrieval_pool_size, retrieval_mode, reranker_variant, *, progress=None, seed=None):
    if progress is not None:
        progress.log("arm_prepare_begin", seed=seed, arm=name, artifact_count=len(package.artifacts), query_count=len(test_items))
    prepare_started = time.perf_counter()
    candidates = package_to_candidates(package, jina_client, embed_model)
    candidate_prepare_s = time.perf_counter() - prepare_started
    query_embed_started = time.perf_counter()
    from scripts.run_stabletoolbench_federated import batched_query_embeddings
    query_embeddings = batched_query_embeddings(jina_client, [item.query for item in test_items], embed_model) if test_items else []
    query_embed_s = time.perf_counter() - query_embed_started
    if progress is not None:
        progress.log("arm_queries_embedded", seed=seed, arm=name, query_count=len(query_embeddings), elapsed_query_embed_s=query_embed_s, total_prepare_s=candidate_prepare_s + query_embed_s)
    candidate_matrix = np.asarray([candidate.embedding for candidate in candidates], dtype=np.float32) if candidates else np.zeros((0, 0), dtype=np.float32)
    if candidate_matrix.size:
        candidate_norms = np.linalg.norm(candidate_matrix, axis=1)
        candidate_norms[candidate_norms == 0.0] = 1.0
        candidate_matrix = candidate_matrix / candidate_norms[:, None]
    rows = []
    total_latency = total_retrieval = total_rerank = 0.0
    inserted_queries = 0
    total_slots_replaced = 0
    for idx, (item, embedding) in enumerate(zip(test_items, query_embeddings), start=1):
        total_started = time.perf_counter()
        retrieval_started = time.perf_counter()
        query_vector = np.asarray(embedding, dtype=np.float32)
        query_norm = float(np.linalg.norm(query_vector))
        if query_norm > 0.0:
            query_vector = query_vector / query_norm
        similarities = candidate_matrix @ query_vector if candidate_matrix.size else np.asarray([], dtype=np.float32)
        ranked, retrieval_pool_tools = build_ranked_candidates(candidates, similarities, retrieval_pool_size, top_k, retrieval_mode)
        oracle_ranked = list(ranked)
        inserted = 0
        gold_missing = [tool for tool in item.gold_tools if tool not in [candidate.parent_tool for candidate in oracle_ranked]]
        if gold_missing:
            tool_to_best = {}
            for cand, score in zip(candidates, similarities.tolist() if candidate_matrix.size else []):
                if cand.parent_tool in gold_missing and cand.parent_tool not in tool_to_best:
                    tool_to_best[cand.parent_tool] = (score, cand)
            replace_idx = len(oracle_ranked) - 1
            for gold_tool in gold_missing:
                cand = tool_to_best.get(gold_tool, (None, None))[1]
                if cand is None or replace_idx < 0:
                    continue
                while replace_idx >= 0 and oracle_ranked[replace_idx].parent_tool in item.gold_tools:
                    replace_idx -= 1
                if replace_idx < 0:
                    break
                oracle_ranked[replace_idx] = cand
                inserted += 1
                replace_idx -= 1
        retrieval_latency = time.perf_counter() - retrieval_started
        if inserted:
            inserted_queries += 1
            total_slots_replaced += inserted
        top_candidate = oracle_ranked[0] if oracle_ranked else None
        if not oracle_ranked or top_candidate is None:
            total_latency_value = time.perf_counter() - total_started
            rows.append({"query_id": item.query_id, "query_text": item.query, "group": item.group, "gold_parent_tools": item.gold_tools, "predicted_tool": "", "routed_correctly": False, "gold_in_top_k": False, "top_candidates": [], "top_candidate_ids": [], "retrieval_pool_tools": retrieval_pool_tools, "parse_ok": False, "fallback_used": True, "latency_seconds": total_latency_value, "retrieval_s": retrieval_latency, "rerank_s": 0.0, "total_s": total_latency_value, "oracle_retrieval": True, "oracle_inserted": False, "oracle_slots_replaced": 0})
            total_latency += total_latency_value
            total_retrieval += retrieval_latency
            continue
        maybe_cuda_synchronize(backend)
        rerank_started = time.perf_counter()
        result = run_prompt(backend, reranker_variant, "toolbench", item.query, oracle_ranked, [], top_candidate)
        maybe_cuda_synchronize(backend)
        rerank_latency = time.perf_counter() - rerank_started
        total_latency_value = time.perf_counter() - total_started
        total_latency += total_latency_value
        total_retrieval += retrieval_latency
        total_rerank += rerank_latency
        rows.append({
            "query_id": item.query_id,
            "query_text": item.query,
            "group": item.group,
            "gold_parent_tools": item.gold_tools,
            "predicted_tool": result.predicted_tool,
            "predicted_candidate": result.predicted_candidate,
            "routed_correctly": result.predicted_tool in item.gold_tools,
            "gold_in_top_k": any(candidate.parent_tool in item.gold_tools for candidate in oracle_ranked),
            "top_candidates": [candidate.parent_tool for candidate in oracle_ranked],
            "top_candidate_ids": [candidate.candidate_id for candidate in oracle_ranked],
            "retrieval_pool_tools": retrieval_pool_tools,
            "parse_ok": result.parse_ok,
            "fallback_used": result.fallback_used,
            "latency_seconds": total_latency_value,
            "retrieval_s": retrieval_latency,
            "rerank_s": rerank_latency,
            "total_s": total_latency_value,
            "prompt_hash": result.prompt_hash,
            "oracle_retrieval": True,
            "oracle_inserted": bool(inserted),
            "oracle_slots_replaced": inserted,
        })
        if progress is not None and (idx == 1 or idx % 25 == 0 or idx == len(test_items)):
            progress.log("arm_progress", seed=seed, arm=name, completed_queries=idx, total_queries=len(test_items), latest_query_id=item.query_id)
    summary = summarize_rows(rows)
    summary.update({
        "arm": name,
        "rows": rows,
        "mean_latency_seconds": total_latency / len(test_items) if test_items else 0.0,
        "mean_retrieval_seconds": total_retrieval / len(test_items) if test_items else 0.0,
        "mean_rerank_seconds": total_rerank / len(test_items) if test_items else 0.0,
        "retrieval_mode": retrieval_mode,
        "retrieval_pool_size": retrieval_pool_size,
        "reranker_variant": reranker_variant,
        "candidate_count": len(candidates),
        "candidate_prepare_s": candidate_prepare_s,
        "query_prepare_s": query_embed_s,
        "oracle_insertion_fraction": inserted_queries / len(test_items) if test_items else 0.0,
        "oracle_mean_slots_replaced": total_slots_replaced / len(test_items) if test_items else 0.0,
    })
    return summary


def aggregate_seed_summaries(seed_summaries):
    grouped = {}
    groups = sorted({group for summary in seed_summaries for group in summary["group_metrics"]})
    for group in groups:
        accs = [summary["group_metrics"][group]["accuracy"] for summary in seed_summaries]
        grouped[group] = {
            "n": seed_summaries[0]["group_metrics"][group]["count"],
            "accuracy": {
                "mean": statistics.mean(accs),
                "sd": statistics.stdev(accs) if len(accs) > 1 else 0.0,
            },
        }
    return {
        "accuracy": {
            "mean": statistics.mean([summary["accuracy"] for summary in seed_summaries]),
            "sd": statistics.stdev([summary["accuracy"] for summary in seed_summaries]) if len(seed_summaries) > 1 else 0.0,
        },
        "oracle_insertion_fraction": {
            "mean": statistics.mean([summary["oracle_insertion_fraction"] for summary in seed_summaries]),
            "sd": statistics.stdev([summary["oracle_insertion_fraction"] for summary in seed_summaries]) if len(seed_summaries) > 1 else 0.0,
        },
        "per_group": grouped,
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    commit, dirty = assert_clean_tree(allow_dirty=args.allow_dirty)
    progress = ProgressLogger(args.output_dir)
    progress.log("launch", git_commit=commit, dirty_entry_count=len(dirty))
    seeds = parse_csv_int(args.seeds)
    arms = parse_csv(args.arms)
    embedder = resolve_local_embedder()
    jina_client = JinaAIClient(api_keys=os.environ.get("JINA_API_KEY") and [os.environ["JINA_API_KEY"]] or [])
    with temporary_env({
        "JINA_LOCAL_EMBED_MODEL": embedder["model_path"],
        "JINA_LOCAL_EMBED_DEVICE": embedder["device"],
        "JINA_LOCAL_EMBED_LOCAL_ONLY": embedder["local_only"],
        "JINA_API_KEY": None,
    }):
        backend = load_local_backend(args.model_path)
        results_by_arm = {arm: [] for arm in arms}
        for seed in seeds:
            progress.log("seed_begin", seed=seed)
            _client_packages, tool_doc_package, built, _aggregator, _merged = build_seed_packages(seed, progress=progress)
            ref_root = A1_R16_ROOTS[seed]
            progress.log("seed_refs_load_begin", seed=seed, ref_root=str(ref_root))
            refs = {arm: load_json(ref_root / f"{arm}.json") for arm in ["synapse", "centralized", "flat_pool"]}
            progress.log("seed_refs_load_done", seed=seed)
            verify_hash("synapse", refs["synapse"]["compendium"]["global_sha256"], built["synapse"][1])
            verify_hash("centralized", refs["centralized"]["compendium"]["global_sha256"], built["centralized"][1])
            verify_hash("flat_pool", refs["flat_pool"]["compendium"]["global_sha256"], built["flat_pool"][1])
            progress.log("seed_query_load_begin", seed=seed, stb_root=str(args.stb_root))
            test_items = load_eval_queries(args.stb_root, ["G1_instruction", "G1_tool", "G1_category", "G2_instruction", "G2_category", "G3_instruction"], tool_doc_package)
            progress.log("seed_query_load_done", seed=seed, query_count=len(test_items))
            for arm in arms:
                package, package_hash = built[arm]
                progress.log("arm_begin", seed=seed, arm=arm, compendium_sha256=package_hash)
                result = evaluate_oracle_arm(arm, package, test_items, jina_client, args.embed_model, backend, args.top_k, args.retrieval_pool_size, args.retrieval_mode, args.reranker_variant, progress=progress, seed=seed)
                result.update({
                    "seed": seed,
                    "arm": arm,
                    "oracle_retrieval": True,
                    "oracle_insertion_rule": "each gold tool absent from top-k replaces the lowest-ranked non-gold candidate, preserving k=5",
                    "compendium_sha256": package_hash,
                    "tool_doc_sha256": built["tool_doc_sha256"],
                    "source_a1_path": str(ref_root / f"{arm}.json"),
                    "git_commit": commit,
                    "embedder": embedder,
                    "paper_eligible": True,
                    "metric_definition": {
                        "correct": "predicted_tool in gold_parent_tools",
                        "recall_at_5": "any gold tool present among candidate tools",
                    },
                })
                results_by_arm[arm].append(result)
                save_json(args.output_dir / f"seed_{seed}" / f"{arm}.json", result)
                progress.log("arm_done", seed=seed, arm=arm, accuracy=result["accuracy"], oracle_insertion_fraction=result["oracle_insertion_fraction"])
        summary = {
            "paper_eligible": True,
            "config": {
                "git_commit": commit,
                "seeds": seeds,
                "arms": arms,
                "oracle_retrieval": True,
                "retrieval_mode": args.retrieval_mode,
                "retrieval_pool_size": args.retrieval_pool_size,
                "top_k": args.top_k,
                "reranker_variant": args.reranker_variant,
                "embed_model": args.embed_model,
                "model_path": args.model_path,
                "local_embedder": embedder,
            },
            "per_arm": {arm: aggregate_seed_summaries(results_by_arm[arm]) for arm in arms},
        }
        save_json(args.output_dir / "summary.json", summary)
        progress.log("complete", output_dir=str(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
