#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_ROOT = Path(os.environ.get("FEDRAG_CANONICAL_ROOT", str(REPO_ROOT))).resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from math_qa import JinaAIClient
from scripts.run_reranker_prompt_sweep import RoutedCandidate, run_prompt
from scripts.run_stabletoolbench_federated import (
    DEFAULT_MODEL_PATH,
    DEFAULT_STB_ROOT,
    GROUPS,
    ProgressLogger,
    apply_junk_filter,
    assert_clean_tree,
    batched_query_embeddings,
    build_tool_registry,
    filter_eval_queries,
    load_local_backend,
    load_package_file,
    load_stabletoolbench_queries,
    maybe_cuda_synchronize,
    package_to_candidates,
    resolve_local_embedder,
    save_json,
    stable_hash,
    summarize_rows,
    temporary_env,
)
from scripts.run_stabletoolbench_heldout import HELDOUT_GROUPS, heldout_tool_set
from synapse.knowledge.compendium import KnowledgePackage


DEFAULT_OUTPUT_DIR = CANONICAL_ROOT / "artifacts" / "results" / "stabletoolbench_candidate_render_interaction_p1_r1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Candidate-source x render-mode replay on persisted E6 compendiums.")
    parser.add_argument("--stb-root", type=Path, default=DEFAULT_STB_ROOT)
    parser.add_argument("--groups", type=str, default=",".join(GROUPS))
    parser.add_argument("--seeds", type=str, default="42,123,456")
    parser.add_argument("--e6-dirs", type=str, required=True, help="Comma-separated seed=dir entries.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retrieval-cap", type=int, default=200)
    parser.add_argument("--candidate-list-size", type=int, default=20)
    parser.add_argument("--reranker-variant", type=str, default="V3")
    parser.add_argument("--embed-model", type=str, default="jina-embeddings-v2-base-en")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args()


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_seed_dirs(value: str) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for entry in parse_csv(value):
        seed, path = entry.split("=", 1)
        out[int(seed)] = Path(path)
    return out


def sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def normalize_matrix(vectors: list[list[float]]) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.size:
        norms = np.linalg.norm(matrix, axis=1)
        norms[norms == 0.0] = 1.0
        matrix = matrix / norms[:, None]
    return matrix


def render_map(candidates: list[RoutedCandidate]) -> dict[str, RoutedCandidate]:
    out: dict[str, RoutedCandidate] = {}
    for candidate in candidates:
        out.setdefault(candidate.parent_tool, candidate)
    return out


def doc_style_candidate(candidate: RoutedCandidate) -> RoutedCandidate:
    # Force the exact docs-only prompt shape: description text only, no scenario
    # or precaution fields.
    description = candidate.when_to_use[0] if candidate.when_to_use else candidate.text
    return replace(candidate, when_to_use=[description], do_not_use_when=[])


def score_query(index: np.ndarray, query_embedding: list[float] | None) -> np.ndarray:
    if not index.size or query_embedding is None:
        return np.asarray([], dtype=np.float32)
    query = np.asarray(query_embedding, dtype=np.float32)
    norm = float(np.linalg.norm(query))
    if norm > 0.0:
        query = query / norm
    return index @ query


def ranked_distinct(candidates: list[RoutedCandidate], scores: np.ndarray, cap: int, top_k: int) -> tuple[list[RoutedCandidate], dict[str, Any], list[RoutedCandidate]]:
    selected: list[RoutedCandidate] = []
    selected_tools: set[str] = set()
    ranked_all: list[RoutedCandidate] = []
    depth_reached = 0
    for depth, idx in enumerate(np.argsort(-scores).tolist(), start=1):
        if depth > cap:
            break
        candidate = candidates[idx]
        if candidate.parent_tool not in {c.parent_tool for c in ranked_all}:
            ranked_all.append(candidate)
        if len(selected) < top_k and candidate.parent_tool not in selected_tools:
            selected_tools.add(candidate.parent_tool)
            selected.append(candidate)
            if len(selected) == top_k:
                depth_reached = depth
    if not depth_reached:
        depth_reached = min(len(candidates), cap)
    diag = {
        "candidate_rule": "distinct5_walkdown",
        "candidate_distinct_count": len({c.parent_tool for c in selected}),
        "candidate_depth_reached": depth_reached,
        "candidate_shortfall": len({c.parent_tool for c in selected}) < top_k,
        "candidate_top_k": top_k,
        "candidate_retrieval_cap": cap,
    }
    return selected, diag, ranked_all


def insert_oracle_gold(
    selected: list[RoutedCandidate],
    ranked_all: list[RoutedCandidate],
    gold_tools: list[str],
    top_k: int,
) -> tuple[list[RoutedCandidate], int]:
    out = list(selected[:top_k])
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


def subset_name(gold_tools: list[str], heldout_tools: set[str]) -> str:
    return "heldout" if gold_tools and all(tool in heldout_tools for tool in gold_tools) else "labeled"


def summarize_subset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    recall = sum(1 for row in rows if row["gold_in_top_k"]) / n if n else 0.0
    accuracy = sum(1 for row in rows if row["correct"]) / n if n else 0.0
    return {
        "n": n,
        "accuracy": accuracy,
        "recall_at_5": recall,
        "conditional_accuracy_given_recall": accuracy / recall if recall else 0.0,
    }


def add_subsets(result: dict[str, Any]) -> None:
    result["subset_metrics"] = {
        "all": summarize_subset(result["rows"]),
        "heldout": summarize_subset([row for row in result["rows"] if row["subset"] == "heldout"]),
        "labeled": summarize_subset([row for row in result["rows"] if row["subset"] == "labeled"]),
    }


def evaluate_cell(
    *,
    seed: int,
    source_name: str,
    render_name: str,
    candidates_by_query: dict[str, list[RoutedCandidate]],
    oracle_candidates_by_query: dict[str, list[RoutedCandidate]] | None,
    render_by_tool: dict[str, RoutedCandidate],
    queries: list[Any],
    heldout_tools: set[str],
    backend: Any,
    args: argparse.Namespace,
    progress: ProgressLogger,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    oracle = oracle_candidates_by_query is not None
    for idx, item in enumerate(queries, start=1):
        started = time.perf_counter()
        base_candidates = (oracle_candidates_by_query or candidates_by_query)[item.query_id]
        rerank_candidates = [render_by_tool[candidate.parent_tool] for candidate in base_candidates]
        if len({candidate.parent_tool for candidate in rerank_candidates}) != args.top_k:
            raise RuntimeError(f"{source_name}/{render_name} seed {seed} query {item.query_id} did not have {args.top_k} distinct render candidates")
        top_candidate = rerank_candidates[0]
        maybe_cuda_synchronize(backend)
        rerank_started = time.perf_counter()
        result = run_prompt(backend, args.reranker_variant, "toolbench", item.query, rerank_candidates, [], top_candidate)
        maybe_cuda_synchronize(backend)
        candidate_tools = [candidate.parent_tool for candidate in base_candidates]
        gold = list(item.gold_tools)
        rows.append(
            {
                "query_id": item.query_id,
                "query_text": item.query,
                "group": item.group,
                "gold_tools": gold,
                "gold_parent_tools": gold,
                "candidate_source": source_name,
                "render_mode": render_name,
                "candidate_tools": candidate_tools,
                "top_candidates": candidate_tools,
                "candidate_ids": [candidate.candidate_id for candidate in base_candidates],
                "render_candidate_ids": [candidate.candidate_id for candidate in rerank_candidates],
                "predicted_tool": result.predicted_tool,
                "predicted_candidate": result.predicted_candidate,
                "correct": result.predicted_tool in gold,
                "routed_correctly": result.predicted_tool in gold,
                "gold_in_top_k": any(tool in gold for tool in candidate_tools),
                "subset": subset_name(gold, heldout_tools),
                "oracle_retrieval": oracle,
                "parse_ok": result.parse_ok,
                "fallback_used": result.fallback_used,
                "rerank_s": time.perf_counter() - rerank_started,
                "total_s": time.perf_counter() - started,
                "prompt_hash": result.prompt_hash,
            }
        )
        if idx == 1 or idx % 50 == 0 or idx == len(queries):
            progress.log(
                "rerank_progress",
                seed=seed,
                candidate_source=source_name,
                render_mode=render_name,
                oracle=oracle,
                completed_queries=idx,
                total_queries=len(queries),
                running_accuracy=sum(row["correct"] for row in rows) / len(rows),
                running_recall_at_5=sum(row["gold_in_top_k"] for row in rows) / len(rows),
            )
    result = summarize_rows(rows)
    result.update(
        {
            "paper_eligible": True,
            "seed": seed,
            "arm": f"{source_name}_candidates_{render_name}_render",
            "candidate_source": source_name,
            "render_mode": render_name,
            "oracle_retrieval": oracle,
            "rows": rows,
        }
    )
    add_subsets(result)
    return result


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for oracle in (False, True):
        bucket = [result for result in results if result["oracle_retrieval"] is oracle]
        key = "oracle" if oracle else "real"
        out[key] = {}
        for arm in sorted({result["arm"] for result in bucket}):
            group = [result for result in bucket if result["arm"] == arm]
            out[key][arm] = {}
            for subset in ("all", "heldout", "labeled"):
                acc = [result["subset_metrics"][subset]["accuracy"] for result in group]
                rec = [result["subset_metrics"][subset]["recall_at_5"] for result in group]
                cond = [result["subset_metrics"][subset]["conditional_accuracy_given_recall"] for result in group]
                out[key][arm][subset] = {
                    "n": statistics.mean([result["subset_metrics"][subset]["n"] for result in group]) if group else 0,
                    "mean_accuracy": statistics.mean(acc) if acc else 0.0,
                    "sd_accuracy": statistics.stdev(acc) if len(acc) > 1 else 0.0,
                    "mean_recall_at_5": statistics.mean(rec) if rec else 0.0,
                    "sd_recall_at_5": statistics.stdev(rec) if len(rec) > 1 else 0.0,
                    "mean_conditional_accuracy_given_recall": statistics.mean(cond) if cond else 0.0,
                    "sd_conditional_accuracy_given_recall": statistics.stdev(cond) if len(cond) > 1 else 0.0,
                    "seeds": [result["seed"] for result in group],
                }
    for key in ("real", "oracle"):
        docs_docs = out[key].get("docs_candidates_docs_render", {}).get("all", {}).get("mean_accuracy")
        docs_comp = out[key].get("docs_candidates_compendium_render", {}).get("all", {}).get("mean_accuracy")
        shared_docs = out[key].get("shared_candidates_docs_render", {}).get("all", {}).get("mean_accuracy")
        shared_comp = out[key].get("shared_candidates_compendium_render", {}).get("all", {}).get("mean_accuracy")
        if None not in (docs_docs, docs_comp, shared_docs, shared_comp):
            out[key]["difference_in_differences_all_accuracy"] = (shared_comp - shared_docs) - (docs_comp - docs_docs)
    return out


def package_hash(package: KnowledgePackage) -> str:
    return stable_hash(
        [
            {
                "signature": artifact.signature,
                "text": artifact.text,
                "metadata": artifact.metadata,
                "structured_payload": artifact.structured_payload,
            }
            for artifact in sorted(package.artifacts, key=lambda artifact: artifact.signature)
        ]
    )


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    commit, dirty = assert_clean_tree(allow_dirty=args.allow_dirty)
    progress = ProgressLogger(args.output_dir)
    progress.log("launch", repo_commit=commit, dirty_entry_count=len(dirty))
    seeds = [int(seed) for seed in parse_csv(args.seeds)]
    seed_dirs = parse_seed_dirs(args.e6_dirs)
    missing = [seed for seed in seeds if seed not in seed_dirs]
    if missing:
        raise SystemExit(f"missing --e6-dirs entries for seeds {missing}")

    groups = parse_csv(args.groups)
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
        registry, junk_info = apply_junk_filter(registry)
        queries, eval_filter = filter_eval_queries(all_queries, registry)
        heldout_tools, heldout_sanity = heldout_tool_set(queries, HELDOUT_GROUPS)
        heldout_set = set(heldout_tools)
        heldout_sha = stable_hash(heldout_tools)
        progress.log("data_ready", test_count=len(queries), heldout_tool_count=len(heldout_tools), heldout_tools_sha256=heldout_sha, junk_filter=junk_info, eval_filter=eval_filter)

        backend = load_local_backend(args.model_path)
        all_results: list[dict[str, Any]] = []
        for seed in seeds:
            package_dir = seed_dirs[seed] / f"seed_{seed}" / "packages"
            flat_package, flat_file_hash = load_package_file(package_dir / "flat_pool.json")
            shared_package, shared_file_hash = load_package_file(package_dir / "synapse_shared.json")
            flat_content_hash = package_hash(flat_package)
            shared_content_hash = package_hash(shared_package)
            docs_candidates = package_to_candidates(flat_package, jina_client, args.embed_model)
            shared_candidates = package_to_candidates(shared_package, jina_client, args.embed_model)
            docs_index = normalize_matrix([candidate.embedding for candidate in docs_candidates])
            shared_index = normalize_matrix([candidate.embedding for candidate in shared_candidates])
            docs_render = {tool: doc_style_candidate(candidate) for tool, candidate in render_map(docs_candidates).items()}
            comp_render = render_map(shared_candidates)
            if set(docs_render) - set(comp_render):
                raise RuntimeError(f"seed {seed} compendium render missing tools from docs package")
            embeddings = batched_query_embeddings(jina_client, [item.query for item in queries], args.embed_model)
            source_configs = {
                "docs": (docs_candidates, docs_index),
                "shared": (shared_candidates, shared_index),
            }
            candidates_by_source: dict[str, dict[str, list[RoutedCandidate]]] = {}
            oracle_by_source: dict[str, dict[str, list[RoutedCandidate]]] = {}
            candidate_records: dict[str, list[dict[str, Any]]] = {}
            oracle_records: dict[str, list[dict[str, Any]]] = {}
            for source_name, (source_candidates, source_index) in source_configs.items():
                candidates_by_query: dict[str, list[RoutedCandidate]] = {}
                oracle_by_query: dict[str, list[RoutedCandidate]] = {}
                records: list[dict[str, Any]] = []
                oracle_rows: list[dict[str, Any]] = []
                for item, embedding in zip(queries, embeddings):
                    scores = score_query(source_index, embedding)
                    selected, diag, ranked_all = ranked_distinct(source_candidates, scores, args.retrieval_cap, args.top_k)
                    if diag["candidate_shortfall"]:
                        raise RuntimeError(f"{source_name} seed {seed} query {item.query_id} short-filled under cap {args.retrieval_cap}")
                    oracle_selected, replacements = insert_oracle_gold(selected, ranked_all, list(item.gold_tools), args.top_k)
                    candidates_by_query[item.query_id] = selected
                    oracle_by_query[item.query_id] = oracle_selected
                    records.append(
                        {
                            "query_id": item.query_id,
                            "group": item.group,
                            "subset": subset_name(list(item.gold_tools), heldout_set),
                            "gold_tools": list(item.gold_tools),
                            "candidate_tools": [candidate.parent_tool for candidate in selected],
                            "candidate_ids": [candidate.candidate_id for candidate in selected],
                            "candidate_top20_tools": [candidate.parent_tool for candidate in ranked_all[: args.candidate_list_size]],
                            "candidate_top20_ids": [candidate.candidate_id for candidate in ranked_all[: args.candidate_list_size]],
                            **diag,
                        }
                    )
                    oracle_rows.append(
                        {
                            "query_id": item.query_id,
                            "gold_tools": list(item.gold_tools),
                            "oracle_candidate_tools": [candidate.parent_tool for candidate in oracle_selected],
                            "oracle_candidate_ids": [candidate.candidate_id for candidate in oracle_selected],
                            "oracle_inserted": replacements > 0,
                            "oracle_replacement_count": replacements,
                            "oracle_insertion_rule": "replace_lowest_ranked_non_gold_preserve_k5",
                        }
                    )
                candidate_hash = sha256_json(records)
                oracle_hash = sha256_json(oracle_rows)
                save_json(args.output_dir / f"seed_{seed}" / source_name / "candidate_lists.json", {"seed": seed, "source": source_name, "rows": records, "sha256": candidate_hash})
                save_json(args.output_dir / f"seed_{seed}" / source_name / "oracle_candidate_lists.json", {"seed": seed, "source": source_name, "rows": oracle_rows, "sha256": oracle_hash})
                candidates_by_source[source_name] = candidates_by_query
                oracle_by_source[source_name] = oracle_by_query
                candidate_records[source_name] = records
                oracle_records[source_name] = oracle_rows
                progress.log("candidate_lists_done", seed=seed, source=source_name, candidate_list_sha256=candidate_hash, oracle_candidate_list_sha256=oracle_hash, mean_depth=sum(row["candidate_depth_reached"] for row in records) / len(records), shortfalls=sum(1 for row in records if row["candidate_shortfall"]))

            for source_name in ("docs", "shared"):
                render_modes = {"docs": docs_render, "compendium": comp_render}
                baseline_tools = {
                    query_id: [candidate.parent_tool for candidate in candidates]
                    for query_id, candidates in candidates_by_source[source_name].items()
                }
                for render_name, render_by_tool in render_modes.items():
                    rendered_tools = {
                        query_id: [candidate.parent_tool for candidate in candidates]
                        for query_id, candidates in candidates_by_source[source_name].items()
                    }
                    if rendered_tools != baseline_tools:
                        raise RuntimeError(f"candidate list changed across render modes for seed {seed} source {source_name}")
                    result = evaluate_cell(
                        seed=seed,
                        source_name=source_name,
                        render_name=render_name,
                        candidates_by_query=candidates_by_source[source_name],
                        oracle_candidates_by_query=None,
                        render_by_tool=render_by_tool,
                        queries=queries,
                        heldout_tools=heldout_set,
                        backend=backend,
                        args=args,
                        progress=progress,
                    )
                    result.update(
                        {
                            "repo_commit": commit,
                            "data_mode": "toolbench_train",
                            "heldout_tools_sha256": heldout_sha,
                            "heldout_sanity": heldout_sanity,
                            "source_package": {
                                "e6_dir": str(seed_dirs[seed]),
                                "flat_pool_file_sha256": flat_file_hash,
                                "synapse_shared_file_sha256": shared_file_hash,
                                "flat_pool_content_sha256": flat_content_hash,
                                "synapse_shared_content_sha256": shared_content_hash,
                            },
                            "candidate_list_sha256": sha256_json(candidate_records[source_name]),
                            "candidate_list_assertion": "byte-identical across render modes within source/query",
                            "metric_definition": {
                                "accuracy": "predicted_tool in gold_tools",
                                "recall_at_5": "any gold tool in the five candidates shown to the reranker",
                            },
                        }
                    )
                    save_json(args.output_dir / f"seed_{seed}" / source_name / f"{render_name}_render.json", result)
                    all_results.append(result)
                    progress.log("cell_done", seed=seed, source=source_name, render=render_name, oracle=False, accuracy=result["accuracy"], recall_at_5=result["recall_at_5"], heldout_acc=result["subset_metrics"]["heldout"]["accuracy"], heldout_r5=result["subset_metrics"]["heldout"]["recall_at_5"])
                    oracle_result = evaluate_cell(
                        seed=seed,
                        source_name=source_name,
                        render_name=render_name,
                        candidates_by_query=candidates_by_source[source_name],
                        oracle_candidates_by_query=oracle_by_source[source_name],
                        render_by_tool=render_by_tool,
                        queries=queries,
                        heldout_tools=heldout_set,
                        backend=backend,
                        args=args,
                        progress=progress,
                    )
                    oracle_result.update(
                        {
                            "repo_commit": commit,
                            "data_mode": "toolbench_train",
                            "heldout_tools_sha256": heldout_sha,
                            "heldout_sanity": heldout_sanity,
                            "source_package": {
                                "e6_dir": str(seed_dirs[seed]),
                                "flat_pool_file_sha256": flat_file_hash,
                                "synapse_shared_file_sha256": shared_file_hash,
                                "flat_pool_content_sha256": flat_content_hash,
                                "synapse_shared_content_sha256": shared_content_hash,
                            },
                            "candidate_list_sha256": sha256_json(candidate_records[source_name]),
                            "candidate_list_assertion": "byte-identical across render modes within source/query",
                            "oracle_candidate_list_sha256": sha256_json(oracle_records[source_name]),
                            "oracle_insertion_rule": "replace_lowest_ranked_non_gold_preserve_k5",
                            "metric_definition": {
                                "accuracy": "predicted_tool in gold_tools",
                                "recall_at_5": "any gold tool in the five candidates shown to the reranker",
                            },
                        }
                    )
                    save_json(args.output_dir / f"seed_{seed}" / source_name / f"{render_name}_render_oracle.json", oracle_result)
                    all_results.append(oracle_result)
                    progress.log("cell_done", seed=seed, source=source_name, render=render_name, oracle=True, accuracy=oracle_result["accuracy"], recall_at_5=oracle_result["recall_at_5"], heldout_acc=oracle_result["subset_metrics"]["heldout"]["accuracy"], heldout_r5=oracle_result["subset_metrics"]["heldout"]["recall_at_5"])
            progress.log("seed_done", seed=seed)

        summary = {
            "paper_eligible": True,
            "config": {
                "repo_commit": commit,
                "seeds": seeds,
                "e6_dirs": {str(seed): str(path) for seed, path in seed_dirs.items()},
                "heldout_tools_sha256": heldout_sha,
                "top_k": args.top_k,
                "retrieval_cap": args.retrieval_cap,
                "candidate_list_size": args.candidate_list_size,
                "reranker_variant": args.reranker_variant,
                "embed_model": args.embed_model,
                "local_embedder": embedder_info,
                "oracle_insertion_rule": "replace_lowest_ranked_non_gold_preserve_k5",
            },
            "aggregate": aggregate(all_results),
        }
        save_json(args.output_dir / "summary.json", summary)
        progress.log("complete", output_dir=str(args.output_dir))
        print(json.dumps(summary["aggregate"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
