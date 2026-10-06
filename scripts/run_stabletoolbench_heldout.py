#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_ROOT = Path(os.environ.get("FEDRAG_CANONICAL_ROOT", str(REPO_ROOT))).resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from math_qa import JinaAIClient
from scripts.run_gsm8k_small_router_sweep import parse_seed_list
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
    candidate_selection_diagnostics,
    combine_packages,
    evaluate_classifier_arm,
    evaluate_local_only_arm,
    evaluate_reranker_arm,
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
    render_precautions,
    resolve_local_embedder,
    save_json,
    stable_hash,
    summarize_rows,
    temporary_env,
    train_query_classifier,
)
from synapse.edge.aggregator import EdgeAggregator, EdgeConfig
from synapse.knowledge.compendium import KnowledgePackage

DEFAULT_OUTPUT_DIR = CANONICAL_ROOT / "artifacts" / "results" / "stabletoolbench_heldout_r1"
HELDOUT_GROUPS = ["G1_tool", "G1_category"]
RERANKER_ARMS = {"synapse", "centralized", "local_only", "flat_pool"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="StableToolBench label-held-out routing experiment.")
    parser.add_argument("--stb-root", type=Path, default=DEFAULT_STB_ROOT)
    parser.add_argument("--toolbench-instruction-dir", type=Path, default=DEFAULT_TOOLBENCH_INSTRUCTION_DIR)
    parser.add_argument("--groups", type=str, default=",".join(GROUPS))
    parser.add_argument("--seeds", type=str, default="42,123,456")
    parser.add_argument("--client-count", type=int, default=5)
    parser.add_argument("--max-items-per-client", type=int, default=5000)
    parser.add_argument("--partition-mode", type=str, default="category", choices=["category", "iid"])
    parser.add_argument("--arms", type=str, default="synapse,centralized,local_only,flat_pool,query_classifier")
    parser.add_argument("--oracle-arms", type=str, default="synapse,flat_pool")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retrieval-pool-size", type=int, default=5)
    parser.add_argument("--retrieval-mode", type=str, default="distinct_tool_topk", choices=["entry_topk", "distinct_tool_topk", "union_doc_scenario"])
    parser.add_argument("--reranker-variant", type=str, default="V3", choices=["V1", "V3"])
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


def sha256_texts(values: list[str]) -> str:
    return hashlib.sha256(json.dumps(values, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()


def normalize_for_name_match(value: str) -> str:
    return " ".join(value.replace("_", " ").replace("-", " ").lower().split())


def heldout_tool_set(all_queries, heldout_groups: list[str]) -> tuple[list[str], dict[str, Any]]:
    tools: set[str] = set()
    sanity: dict[str, Any] = {}
    for group in heldout_groups:
        group_queries = [query for query in all_queries if query.group == group]
        group_tools = {tool for query in group_queries for tool in query.gold_tools}
        tools.update(group_tools)
        sanity[group] = {
            "query_count": len(group_queries),
            "queries_with_gold_in_heldout": len(group_queries),
            "distinct_gold_tools": len(group_tools),
        }
    heldout = sorted(tools)
    for group in heldout_groups:
        group_queries = [query for query in all_queries if query.group == group]
        sanity[group]["queries_with_gold_in_heldout"] = sum(
            1 for query in group_queries if any(tool in tools for tool in query.gold_tools)
        )
    return heldout, sanity


def item_mentions_heldout_tool(item, heldout_tools: set[str]) -> bool:
    text_parts = [item.query, *(tool for tool in item.gold_tools)]
    for api in item.api_list:
        text_parts.extend(str(api.get(key) or "") for key in ("tool_name", "api_name", "category_name", "api_description"))
    normalized = normalize_for_name_match(" ".join(text_parts))
    return any(normalize_for_name_match(tool) in normalized for tool in heldout_tools)


def filter_heldout_pool(items, heldout_tools: set[str]) -> tuple[list[Any], dict[str, Any]]:
    kept = []
    dropped_label = []
    dropped_mention = []
    for item in items:
        if any(tool in heldout_tools for tool in item.gold_tools):
            dropped_label.append(item)
            continue
        if item_mentions_heldout_tool(item, heldout_tools):
            dropped_mention.append(item)
            continue
        kept.append(item)
    remaining_label_hits = sum(1 for item in kept if any(tool in heldout_tools for tool in item.gold_tools))
    return kept, {
        "pool_items_before": len(items),
        "pool_items_dropped_heldout": len(dropped_label),
        "experience_entries_dropped_heldout": len(dropped_label) + len(dropped_mention),
        "experience_entries_dropped_heldout_mentions": len(dropped_mention),
        "pool_items_after": len(kept),
        "remaining_items_with_heldout_label": remaining_label_hits,
        "dropped_heldout_query_ids_sample": [item.query_id for item in (dropped_label + dropped_mention)[:20]],
    }


def assign_clients(items: list[Any], client_count: int, partition_mode: str, seed: int) -> dict[str, list[Any]]:
    # Import lazily to keep this runner's public imports explicit above.
    from scripts.run_stabletoolbench_federated import assign_clients as _assign_clients

    return _assign_clients(items, client_count, partition_mode, seed)


def subset_name(query, heldout_tools: set[str]) -> str:
    return "heldout" if query.gold_tools and all(tool in heldout_tools for tool in query.gold_tools) else "labeled"


def add_subset_fields(result: dict[str, Any], heldout_tools: set[str]) -> dict[str, Any]:
    for row in result.get("rows", []):
        row["subset"] = "heldout" if row.get("gold_parent_tools") and all(tool in heldout_tools for tool in row.get("gold_parent_tools", [])) else "labeled"
        row["gold_tools"] = row.get("gold_parent_tools", [])
        row["correct"] = bool(row.get("routed_correctly"))
    result["subset_metrics"] = summarize_subsets(result.get("rows", []), heldout_tools)
    return result


def summarize_subsets(rows: list[dict[str, Any]], heldout_tools: set[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for subset in ("heldout", "labeled"):
        subset_rows = [row for row in rows if row.get("subset") == subset]
        out[subset] = summarize_row_subset(subset_rows)
    heldout_rows = [row for row in rows if row.get("subset") == "heldout"]
    out["heldout_by_group"] = {}
    for group in sorted({row.get("group") for row in heldout_rows}):
        out["heldout_by_group"][str(group)] = summarize_row_subset([row for row in heldout_rows if row.get("group") == group])
    out["heldout_definition"] = "all gold_parent_tools are in H"
    out["labeled_definition"] = "at least one gold_parent_tool is outside H"
    out["heldout_tool_count"] = len(heldout_tools)
    return out


def summarize_row_subset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "accuracy": sum(1 for row in rows if row.get("routed_correctly")) / len(rows) if rows else 0.0,
        "recall_at_5": sum(1 for row in rows if row.get("gold_in_top_k")) / len(rows) if rows else 0.0,
        "retrieval_top1": sum(
            1
            for row in rows
            if (row.get("top_candidates") or [""])[0] in set(row.get("gold_parent_tools") or [])
        ) / len(rows) if rows else 0.0,
    }


def oracle_ranked(ranked_topk: list[Any], ranked_all: list[Any], gold_tools: list[str], top_k: int) -> tuple[list[Any], int]:
    if not ranked_all:
        return ranked_topk, 0
    out = list(ranked_topk[:top_k])
    tools = [candidate.parent_tool for candidate in out]
    replacements = 0
    for gold_tool in gold_tools:
        if gold_tool in tools:
            continue
        gold_candidate = next((candidate for candidate in ranked_all if candidate.parent_tool == gold_tool), None)
        if gold_candidate is None:
            continue
        replace_index = None
        for idx in range(len(out) - 1, -1, -1):
            if out[idx].parent_tool not in gold_tools:
                replace_index = idx
                break
        if replace_index is None:
            continue
        out[replace_index] = gold_candidate
        tools[replace_index] = gold_candidate.parent_tool
        replacements += 1
    return out, replacements


def evaluate_oracle_arm(
    name: str,
    package: KnowledgePackage,
    test_items,
    heldout_tools: set[str],
    jina_client: JinaAIClient,
    embed_model: str,
    backend: Any,
    top_k: int,
    retrieval_pool_size: int,
    retrieval_mode: str,
    reranker_variant: str,
    *,
    progress: ProgressLogger,
    seed: int,
) -> dict[str, Any]:
    heldout_items = [item for item in test_items if subset_name(item, heldout_tools) == "heldout"]
    progress.log("oracle_prepare_begin", seed=seed, arm=name, query_count=len(heldout_items), artifact_count=len(package.artifacts))
    prepare_started = time.perf_counter()
    candidates = package_to_candidates(package, jina_client, embed_model)
    candidate_matrix = np.asarray([candidate.embedding for candidate in candidates], dtype=np.float32) if candidates else np.zeros((0, 0), dtype=np.float32)
    if candidate_matrix.size:
        norms = np.linalg.norm(candidate_matrix, axis=1)
        norms[norms == 0.0] = 1.0
        candidate_matrix = candidate_matrix / norms[:, None]
    query_embeddings = batched_query_embeddings(jina_client, [item.query for item in heldout_items], embed_model) if heldout_items else []
    progress.log("oracle_prepare_done", seed=seed, arm=name, elapsed_prepare_s=time.perf_counter() - prepare_started)
    rows = []
    inserted_rows = 0
    replacement_total = 0
    for idx, (item, embedding) in enumerate(zip(heldout_items, query_embeddings), start=1):
        started = time.perf_counter()
        q = np.asarray(embedding, dtype=np.float32)
        qn = float(np.linalg.norm(q))
        if qn > 0.0:
            q = q / qn
        similarities = candidate_matrix @ q if candidate_matrix.size else np.asarray([], dtype=np.float32)
        initial_ranked, retrieval_pool_tools = build_ranked_candidates(candidates, similarities, retrieval_pool_size, top_k, retrieval_mode)
        initial_diag = candidate_selection_diagnostics(candidates, similarities, initial_ranked, top_k, retrieval_pool_size, retrieval_mode)
        if initial_diag["candidate_shortfall"] and retrieval_mode == "distinct_tool_topk":
            raise RuntimeError(
                f"{name}_oracle seed {seed} query {item.query_id} returned "
                f"{initial_diag['candidate_distinct_count']} distinct candidates under cap {retrieval_pool_size}"
            )
        ranked_all = [candidates[idx] for idx in np.argsort(-similarities)] if len(similarities) else []
        ranked, replacements = oracle_ranked(initial_ranked, ranked_all, item.gold_tools, top_k)
        candidate_diag = candidate_selection_diagnostics(candidates, similarities, ranked, top_k, retrieval_pool_size, retrieval_mode)
        inserted_rows += int(replacements > 0)
        replacement_total += replacements
        top_candidate = ranked[0] if ranked else None
        if top_candidate is None:
            result_tool = ""
            prompt_hash = ""
            parse_ok = False
            fallback = True
            rerank_s = 0.0
        else:
            maybe_cuda_synchronize(backend)
            rerank_started = time.perf_counter()
            result = run_prompt(backend, reranker_variant, "toolbench", item.query, ranked, [], top_candidate)
            maybe_cuda_synchronize(backend)
            rerank_s = time.perf_counter() - rerank_started
            result_tool = result.predicted_tool
            prompt_hash = result.prompt_hash
            parse_ok = result.parse_ok
            fallback = result.fallback_used
        rows.append({
            "query_id": item.query_id,
            "query_text": item.query,
            "group": item.group,
            "gold_parent_tools": item.gold_tools,
            "gold_tools": item.gold_tools,
            "subset": "heldout",
            "predicted_tool": result_tool,
            "routed_correctly": result_tool in item.gold_tools,
            "correct": result_tool in item.gold_tools,
            "oracle_retrieval": True,
            "oracle_inserted": replacements > 0,
            "oracle_replacement_count": replacements,
            "gold_in_top_k": any(candidate.parent_tool in item.gold_tools for candidate in ranked),
            "top_candidates": [candidate.parent_tool for candidate in ranked],
            "top_candidate_ids": [candidate.candidate_id for candidate in ranked],
            "retrieval_pool_tools": retrieval_pool_tools,
            **candidate_diag,
            "pre_oracle_candidate_distinct_count": initial_diag["candidate_distinct_count"],
            "pre_oracle_candidate_depth_reached": initial_diag["candidate_depth_reached"],
            "pre_oracle_candidate_shortfall": initial_diag["candidate_shortfall"],
            "parse_ok": parse_ok,
            "fallback_used": fallback,
            "retrieval_s": 0.0,
            "rerank_s": rerank_s,
            "total_s": time.perf_counter() - started,
            "prompt_hash": prompt_hash,
        })
        if idx == 1 or idx % 25 == 0 or idx == len(heldout_items):
            progress.log("oracle_progress", seed=seed, arm=name, completed_queries=idx, total_queries=len(heldout_items), running_accuracy=sum(1 for row in rows if row["routed_correctly"]) / len(rows))
    summary = summarize_rows(rows)
    summary.update({
        "arm": f"{name}_oracle",
        "rows": rows,
        "oracle_retrieval": True,
        "oracle_insertion_fraction": inserted_rows / len(rows) if rows else 0.0,
        "oracle_replacement_mean": replacement_total / len(rows) if rows else 0.0,
        "subset_metrics": summarize_subsets(rows, heldout_tools),
    })
    return summary


def aggregate_subset_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    arms = sorted({result["arm"] for result in results})
    for arm in arms:
        out[arm] = {}
        arm_results = [result for result in results if result["arm"] == arm]
        for subset in ("heldout", "labeled"):
            acc = [result["subset_metrics"][subset]["accuracy"] for result in arm_results if subset in result.get("subset_metrics", {})]
            rec = [result["subset_metrics"][subset]["recall_at_5"] for result in arm_results if subset in result.get("subset_metrics", {})]
            top1 = [result["subset_metrics"][subset]["retrieval_top1"] for result in arm_results if subset in result.get("subset_metrics", {})]
            n = [result["subset_metrics"][subset]["n"] for result in arm_results if subset in result.get("subset_metrics", {})]
            out[arm][subset] = {
                "mean_accuracy": statistics.mean(acc) if acc else 0.0,
                "sd_accuracy": statistics.stdev(acc) if len(acc) > 1 else 0.0,
                "mean_recall_at_5": statistics.mean(rec) if rec else 0.0,
                "sd_recall_at_5": statistics.stdev(rec) if len(rec) > 1 else 0.0,
                "mean_retrieval_top1": statistics.mean(top1) if top1 else 0.0,
                "sd_retrieval_top1": statistics.stdev(top1) if len(top1) > 1 else 0.0,
                "n_per_seed": n,
            }
        out[arm]["heldout_by_group"] = {}
        for group in HELDOUT_GROUPS:
            vals = [
                result["subset_metrics"].get("heldout_by_group", {}).get(group, {}).get("accuracy")
                for result in arm_results
            ]
            vals = [float(value) for value in vals if value is not None]
            out[arm]["heldout_by_group"][group] = {
                "mean_accuracy": statistics.mean(vals) if vals else 0.0,
                "sd_accuracy": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            }
    return out


def mcnemar(rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]]) -> dict[str, int]:
    by_b = {row["query_id"]: row for row in rows_b}
    a_only = b_only = same = compared = 0
    for row in rows_a:
        other = by_b.get(row["query_id"])
        if other is None or row.get("subset") != "heldout":
            continue
        compared += 1
        ac = bool(row.get("routed_correctly"))
        bc = bool(other.get("routed_correctly"))
        if ac and not bc:
            a_only += 1
        elif bc and not ac:
            b_only += 1
        else:
            same += 1
    return {"compared": compared, "synapse_correct_flat_wrong": a_only, "flat_correct_synapse_wrong": b_only, "same": same}


def main() -> int:
    load_dotenv(REPO_ROOT / ".env")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    commit, dirty_entries = assert_clean_tree(allow_dirty=args.allow_dirty)
    progress = ProgressLogger(args.output_dir)
    groups = parse_csv(args.groups)
    arms = parse_csv(args.arms)
    oracle_arms = parse_csv(args.oracle_arms)
    seeds = parse_seed_list(args.seeds)
    progress.log("start", git_commit=commit, groups=groups, arms=arms, oracle_arms=oracle_arms, seeds=seeds)

    queries = load_stabletoolbench_queries(args.stb_root, groups)
    progress.log("load_training_begin", instruction_dir=str(args.toolbench_instruction_dir))
    train_items = load_toolbench_training_items(args.toolbench_instruction_dir)
    base_registry = build_tool_registry(queries + train_items)
    progress.log("load_training_done", train_count=len(train_items), registry_tool_count=len(base_registry))

    base_registry, registry_filter_info = apply_junk_filter(base_registry)
    queries, query_filter_info = filter_eval_queries(queries, base_registry)
    train_items, train_filter_info = filter_experience_items(train_items, base_registry)
    progress.log("junk_filter_done", registry_tool_count=len(base_registry), query_count=len(queries), train_count=len(train_items), **registry_filter_info, **query_filter_info, **train_filter_info)

    heldout_tools, heldout_sanity = heldout_tool_set(queries, HELDOUT_GROUPS)
    heldout_sha = sha256_texts(heldout_tools)
    heldout_path = args.output_dir / "heldout_tools.json"
    save_json(heldout_path, {"heldout_tools": heldout_tools, "heldout_tools_sha256": heldout_sha, "sanity": heldout_sanity})
    progress.log("heldout_tools_ready", heldout_tool_count=len(heldout_tools), heldout_tools_sha256=heldout_sha, sanity=heldout_sanity)

    embedder_info = resolve_local_embedder()
    jina_client = JinaAIClient(api_keys=[])
    with temporary_env({
        "JINA_LOCAL_EMBED_MODEL": embedder_info["model_path"],
        "JINA_LOCAL_EMBED_DEVICE": embedder_info["device"],
        "JINA_LOCAL_EMBED_LOCAL_ONLY": embedder_info["local_only"],
        "JINA_API_KEY": None,
    }):
        train_items, exact_overlap_info = remove_exact_eval_overlaps(train_items, queries)
        train_items, near_overlap_info, forbidden_near_duplicate_texts = remove_near_duplicate_eval_overlaps(
            train_items,
            queries,
            jina_client,
            args.embed_model,
            removal_threshold=args.contamination_near_duplicate_threshold,
            report_threshold=args.contamination_report_threshold,
            pool_embedding_cache_dir=args.pool_embedding_cache_dir,
            progress=progress,
        )
        near_overlap_info.pop("pool_items_removed_near_dup_query_ids", None)
        leak_filter = {**exact_overlap_info, **near_overlap_info}
        leak_filter["pool_items_removed_total"] = int(leak_filter["pool_items_removed_exact"]) + int(leak_filter["pool_items_removed_near_dup"])
        leak_filter["eval_queries_removed"] = 0
        leak_filter["post_filter_refusal_check"] = assert_no_eval_overlap(train_items, queries, forbidden_near_duplicate_texts=forbidden_near_duplicate_texts, stage="e5_post_leak_filter_pool")
        progress.log("contamination_filter_done", train_count=len(train_items), test_count=len(queries), **leak_filter)

        heldout_set = set(heldout_tools)
        train_items, heldout_filter = filter_heldout_pool(train_items, heldout_set)
        if heldout_filter["remaining_items_with_heldout_label"] != 0:
            raise RuntimeError("heldout label filter failed")
        heldout_filter["eval_queries_removed"] = 0
        heldout_filter["heldout_tools_sha256"] = heldout_sha
        heldout_filter["pool_sha256"] = stable_hash([{"query_id": item.query_id, "query": item.query, "gold_tools": item.gold_tools} for item in train_items])
        progress.log("heldout_pool_filter_done", **heldout_filter)

        backend = None
        if any(arm in RERANKER_ARMS for arm in arms) or oracle_arms:
            progress.log("load_backend_begin", model_path=args.model_path)
            backend = load_local_backend(args.model_path)
            progress.log("load_backend_done")

        registry, doc_drift_info = base_registry, {"enabled": False, "mode": "none", "fraction": 0.0, "drifted_tools": []}
        all_results: list[dict[str, Any]] = []
        paired: dict[str, Any] = {}
        for seed in seeds:
            progress.log("seed_begin", seed=seed)
            clients = assign_clients(train_items, args.client_count, args.partition_mode, seed)
            original_client_sizes = {client_id: len(items) for client_id, items in sorted(clients.items())}
            clients = limit_client_items(clients, args.max_items_per_client, seed)
            client_fit_items = [item for items in clients.values() for item in items]
            if any(tool in heldout_set for item in client_fit_items for tool in item.gold_tools):
                raise RuntimeError(f"heldout labels leaked into seed {seed} client fit set")
            progress.log("client_partition_done", seed=seed, original_client_sizes=original_client_sizes, capped_client_sizes={client_id: len(items) for client_id, items in sorted(clients.items())}, classifier_train_n=len(client_fit_items))
            client_packages = {}
            client_hashes = []
            oracle_packages: dict[str, tuple[KnowledgePackage, str]] = {}
            seed_dir = args.output_dir / f"seed_{seed}"
            for client_id, items in sorted(clients.items()):
                progress.log("client_package_begin", seed=seed, client_id=client_id, item_count=len(items))
                package, package_hash = build_client_package(client_id, items, registry, jina_client, args.embed_model)
                client_packages[client_id] = package
                client_hashes.append(package_hash)
                save_json(seed_dir / "packages" / f"{client_id}.json", {"package_sha256": package_hash, "package": {"source_id": package.source_id, "metadata": package.metadata, "artifact_count": len(package.artifacts)}})
                progress.log("client_package_done", seed=seed, client_id=client_id, artifact_count=len(package.artifacts), package_sha256=package_hash)
            tool_doc_package, tool_doc_hash = build_doc_package(registry)
            flat_package, flat_hash = build_flat_pool_package(registry)
            classifier_bundle = train_query_classifier(client_fit_items) if "query_classifier" in arms else None
            if classifier_bundle is not None:
                classifier_classes = set(classifier_bundle.classes)
                if classifier_classes & heldout_set:
                    raise RuntimeError(f"classifier label space contains heldout tools: {sorted(classifier_classes & heldout_set)[:10]}")
            common = {
                "paper_eligible": True,
                "repo_commit": commit,
                "seed": seed,
                "data_mode": "toolbench_train",
                "heldout_tools_sha256": heldout_sha,
                "heldout_tool_count": len(heldout_tools),
                "pool_sha256": heldout_filter["pool_sha256"],
                "heldout_filter": heldout_filter,
                "contamination_filter": leak_filter,
                "eval_queries_removed": 0,
                "metric_definition": {"correct": "predicted_tool in gold_parent_tools", "heldout": "all gold tools in H"},
                "embedder": embedder_info,
                "doc_drift": doc_drift_info,
            }
            seed_results: dict[str, dict[str, Any]] = {}
            if "synapse" in arms:
                progress.log("arm_merge_begin", seed=seed, arm="synapse", merge_policy=args.merge_policy)
                with temporary_env({"SYNAPSE_EDGE_MERGE_POLICY": args.merge_policy}):
                    aggregator = EdgeAggregator(EdgeConfig(edge_id=f"stabletoolbench_heldout_seed_{seed}"))
                    merged = aggregator.merge_packages(list(client_packages.values()))
                if merged is None:
                    raise RuntimeError(f"Seed {seed} synapse merge produced no package")
                synapse_package, synapse_hash = combine_packages("synapse_with_docs", [tool_doc_package, merged])
                result = evaluate_reranker_arm("synapse", synapse_package, queries, jina_client, args.embed_model, backend, args.top_k, args.retrieval_pool_size, args.retrieval_mode, args.reranker_variant, progress=progress, seed=seed)
                result.update(common)
                result.update({"arm": "synapse", "compendium": {"global_sha256": synapse_hash, "client_sha256": client_hashes, "tool_doc_sha256": tool_doc_hash, "artifact_count": len(synapse_package.artifacts)}})
                add_subset_fields(result, heldout_set)
                save_json(seed_dir / "synapse.json", result)
                all_results.append(result)
                seed_results["synapse"] = result
                oracle_packages["synapse"] = (synapse_package, synapse_hash)
                progress.log("arm_done", seed=seed, arm="synapse", accuracy=result["accuracy"], heldout_accuracy=result["subset_metrics"]["heldout"]["accuracy"])
            if "centralized" in arms:
                pooled, pooled_hash = combine_packages("centralized_experience", list(client_packages.values()))
                package, package_hash = combine_packages("centralized_with_docs", [tool_doc_package, pooled])
                result = evaluate_reranker_arm("centralized", package, queries, jina_client, args.embed_model, backend, args.top_k, args.retrieval_pool_size, args.retrieval_mode, args.reranker_variant, progress=progress, seed=seed)
                result.update(common)
                result.update({"arm": "centralized", "compendium": {"global_sha256": package_hash, "pooled_experience_sha256": pooled_hash, "client_sha256": client_hashes, "tool_doc_sha256": tool_doc_hash, "artifact_count": len(package.artifacts)}})
                add_subset_fields(result, heldout_set)
                save_json(seed_dir / "centralized.json", result)
                all_results.append(result)
                seed_results["centralized"] = result
                progress.log("arm_done", seed=seed, arm="centralized", accuracy=result["accuracy"], heldout_accuracy=result["subset_metrics"]["heldout"]["accuracy"])
            if "local_only" in arms:
                result = evaluate_local_only_arm(client_packages, tool_doc_package, queries, jina_client, args.embed_model, backend, args.top_k, args.retrieval_pool_size, args.retrieval_mode, args.reranker_variant, progress=progress, seed=seed)
                result.update(common)
                result.update({"arm": "local_only", "compendium": {"client_sha256": client_hashes, "tool_doc_sha256": tool_doc_hash}})
                add_subset_fields(result, heldout_set)
                save_json(seed_dir / "local_only.json", result)
                all_results.append(result)
                seed_results["local_only"] = result
                progress.log("arm_done", seed=seed, arm="local_only", accuracy=result["accuracy"], heldout_accuracy=result["subset_metrics"]["heldout"]["accuracy"])
            if "flat_pool" in arms:
                result = evaluate_reranker_arm("flat_pool", flat_package, queries, jina_client, args.embed_model, backend, args.top_k, args.retrieval_pool_size, args.retrieval_mode, args.reranker_variant, progress=progress, seed=seed)
                result.update(common)
                result.update({"arm": "flat_pool", "compendium": {"global_sha256": flat_hash, "artifact_count": len(flat_package.artifacts)}})
                add_subset_fields(result, heldout_set)
                save_json(seed_dir / "flat_pool.json", result)
                all_results.append(result)
                seed_results["flat_pool"] = result
                oracle_packages["flat_pool"] = (flat_package, flat_hash)
                progress.log("arm_done", seed=seed, arm="flat_pool", accuracy=result["accuracy"], heldout_accuracy=result["subset_metrics"]["heldout"]["accuracy"])
            if "query_classifier" in arms:
                result = evaluate_classifier_arm(classifier_bundle, queries)
                result.update(common)
                result.update({"arm": "query_classifier", "classifier_train_n": len(client_fit_items), "classifier_heldout_predictions_in_H": sum(1 for row in result["rows"] if subset_name(row_to_query_like(row), heldout_set) == "heldout" and row.get("predicted_tool") in heldout_set)})
                add_subset_fields(result, heldout_set)
                if result["classifier_heldout_predictions_in_H"] != 0:
                    raise RuntimeError(f"classifier predicted heldout tools on heldout queries for seed {seed}")
                save_json(seed_dir / "query_classifier.json", result)
                all_results.append(result)
                seed_results["query_classifier"] = result
                progress.log("arm_done", seed=seed, arm="query_classifier", accuracy=result["accuracy"], heldout_accuracy=result["subset_metrics"]["heldout"]["accuracy"])
            for arm in oracle_arms:
                if arm not in oracle_packages:
                    continue
                package, package_hash = oracle_packages[arm]
                result = evaluate_oracle_arm(arm, package, queries, heldout_set, jina_client, args.embed_model, backend, args.top_k, args.retrieval_pool_size, args.retrieval_mode, args.reranker_variant, progress=progress, seed=seed)
                result.update(common)
                result.update({"compendium": {"global_sha256": package_hash, "artifact_count": len(package.artifacts)}})
                save_json(seed_dir / f"{arm}_oracle.json", result)
                all_results.append(result)
                progress.log("oracle_done", seed=seed, arm=arm, accuracy=result["accuracy"], oracle_insertion_fraction=result["oracle_insertion_fraction"])
            if "synapse" in seed_results and "flat_pool" in seed_results:
                paired[str(seed)] = mcnemar(seed_results["synapse"]["rows"], seed_results["flat_pool"]["rows"])
            progress.log("seed_complete", seed=seed)
    summary = {
        "paper_eligible": True,
        "config": {
            "repo_commit": commit,
            "seeds": seeds,
            "groups": groups,
            "heldout_groups": HELDOUT_GROUPS,
            "heldout_tools_sha256": heldout_sha,
            "heldout_tools_path": str(heldout_path),
            "arms": arms,
            "oracle_arms": oracle_arms,
            "client_count": args.client_count,
            "max_items_per_client": args.max_items_per_client,
            "partition_mode": args.partition_mode,
            "top_k": args.top_k,
            "retrieval_pool_size": args.retrieval_pool_size,
            "candidate_rule": "distinct5_walkdown" if args.retrieval_mode == "distinct_tool_topk" else args.retrieval_mode,
            "candidate_shortfall_refusal": args.retrieval_mode == "distinct_tool_topk",
            "retrieval_mode": args.retrieval_mode,
            "reranker_variant": args.reranker_variant,
            "merge_policy": args.merge_policy,
            "local_embedder": embedder_info,
        },
        "heldout_sanity": heldout_sanity,
        "heldout_filter": heldout_filter,
        "aggregate": aggregate_subset_results(all_results),
        "paired_mcnemar_synapse_vs_flat_pool": paired,
    }
    save_json(args.output_dir / "summary.json", summary)
    progress.log("complete", output_dir=str(args.output_dir))
    print(json.dumps(summary["aggregate"], indent=2))
    return 0


def row_to_query_like(row: dict[str, Any]):
    return type("RowQuery", (), {"gold_tools": row.get("gold_parent_tools", [])})


if __name__ == "__main__":
    raise SystemExit(main())
