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
CANONICAL_ROOT = Path(os.environ.get("FEDRAG_CANONICAL_ROOT", str(REPO_ROOT))).resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from math_qa import JinaAIClient
from scripts.run_reranker_prompt_sweep import run_prompt
from scripts.run_stabletoolbench_category_holdout_e7 import (
    DEFAULT_INPUT_DIR,
    item_record,
    parse_csv,
    parse_ints,
    subset_for,
    subset_metrics,
)
from scripts.run_stabletoolbench_federated import (
    DEFAULT_MODEL_PATH,
    DEFAULT_STB_ROOT,
    DEFAULT_TOOLBENCH_INSTRUCTION_DIR,
    GROUPS,
    ProgressLogger,
    apply_junk_filter,
    assign_clients,
    assert_clean_tree,
    batched_query_embeddings,
    build_client_package,
    build_doc_package,
    build_ranked_candidates,
    build_tool_registry,
    combine_packages,
    filter_eval_queries,
    filter_experience_items,
    limit_client_items,
    load_local_backend,
    load_stabletoolbench_queries,
    load_toolbench_training_items,
    maybe_cuda_synchronize,
    package_to_candidates,
    resolve_local_embedder,
    save_json,
    stable_hash,
    summarize_rows,
    temporary_env,
)
from scripts.run_stabletoolbench_heldout import filter_heldout_pool, sha256_texts


DEFAULT_OUTPUT_DIR = CANONICAL_ROOT / "artifacts" / "results" / "stabletoolbench_category_holdout_e7_two_index_r1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E7-b category hold-out two-index slot-budget router.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stb-root", type=Path, default=DEFAULT_STB_ROOT)
    parser.add_argument("--toolbench-instruction-dir", type=Path, default=DEFAULT_TOOLBENCH_INSTRUCTION_DIR)
    parser.add_argument("--groups", default=",".join(GROUPS))
    parser.add_argument("--seeds", default="42,123,456")
    parser.add_argument("--client-count", type=int, default=5)
    parser.add_argument("--max-items-per-client", type=int, default=5000)
    parser.add_argument("--partition-mode", choices=("category", "iid"), default="category")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--docs-budget", type=int, default=3)
    parser.add_argument("--retrieval-pool-size", type=int, default=5000)
    parser.add_argument("--reranker-variant", default="V3")
    parser.add_argument("--embed-model", default="jina-embeddings-v2-base-en")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args()


def normalize(vectors: list[list[float]]) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.size:
        norms = np.linalg.norm(matrix, axis=1)
        norms[norms == 0.0] = 1.0
        matrix = matrix / norms[:, None]
    return matrix


def score(matrix: np.ndarray, query_embedding: list[float]) -> np.ndarray:
    q = np.asarray(query_embedding, dtype=np.float32)
    norm = float(np.linalg.norm(q))
    if norm > 0.0:
        q = q / norm
    return matrix @ q if getattr(matrix, "size", 0) else np.asarray([], dtype=np.float32)


def top_distinct(candidates: list[Any], scores: np.ndarray, pool_size: int) -> list[Any]:
    ranked, _ = build_ranked_candidates(candidates, scores, pool_size, pool_size, "distinct_tool_topk")
    return ranked


def fuse_budget(docs: list[Any], exp: list[Any], top_k: int, docs_budget: int) -> tuple[list[Any], list[str]]:
    ordered: list[Any] = []
    sources: list[str] = []
    seen: set[str] = set()

    def add(candidate: Any, source: str) -> bool:
        if candidate.parent_tool in seen:
            return False
        seen.add(candidate.parent_tool)
        ordered.append(candidate)
        sources.append(source)
        return len(ordered) == top_k

    docs_added = 0
    for candidate in docs:
        if add(candidate, "docs"):
            return ordered, sources
        docs_added += 1
        if docs_added >= docs_budget:
            break
    for candidate in exp:
        if add(candidate, "experience"):
            return ordered, sources
    for candidate in docs:
        if add(candidate, "docs_backfill"):
            return ordered, sources
    return ordered, sources


def add_subsets(result: dict[str, Any]) -> None:
    result["subset_metrics"] = {}
    for subset in ("heldout", "partial_heldout", "labeled"):
        result["subset_metrics"][subset] = subset_metrics([row for row in result["rows"] if row.get("subset") == subset])


def evaluate_two_index(
    *,
    seed: int,
    queries: list[Any],
    heldout_tools: set[str],
    docs_pkg: Any,
    exp_pkg: Any,
    rerank_pkg: Any,
    jina: JinaAIClient,
    backend: Any,
    args: argparse.Namespace,
    progress: ProgressLogger,
) -> dict[str, Any]:
    docs_candidates = package_to_candidates(docs_pkg, jina, args.embed_model)
    exp_candidates = package_to_candidates(exp_pkg, jina, args.embed_model)
    rerank_candidates = package_to_candidates(rerank_pkg, jina, args.embed_model)
    rerank_by_tool: dict[str, Any] = {}
    for candidate in rerank_candidates:
        rerank_by_tool.setdefault(candidate.parent_tool, candidate)

    docs_index = normalize([candidate.embedding for candidate in docs_candidates])
    exp_index = normalize([candidate.embedding for candidate in exp_candidates])
    query_embeddings = batched_query_embeddings(jina, [item.query for item in queries], args.embed_model)

    rows: list[dict[str, Any]] = []
    shortfalls = 0
    for idx, (item, query_embedding) in enumerate(zip(queries, query_embeddings), start=1):
        started = time.perf_counter()
        docs_ranked = top_distinct(docs_candidates, score(docs_index, query_embedding), args.retrieval_pool_size)
        exp_ranked = top_distinct(exp_candidates, score(exp_index, query_embedding), args.retrieval_pool_size)
        fused, candidate_sources = fuse_budget(docs_ranked, exp_ranked, args.top_k, args.docs_budget)
        if len(fused) < args.top_k:
            shortfalls += 1
        rerank_candidates_for_query = [rerank_by_tool.get(candidate.parent_tool, candidate) for candidate in fused]
        maybe_cuda_synchronize(backend)
        rerank_started = time.perf_counter()
        prompt = run_prompt(
            backend,
            args.reranker_variant,
            "toolbench",
            item.query,
            rerank_candidates_for_query,
            [],
            rerank_candidates_for_query[0],
        )
        maybe_cuda_synchronize(backend)
        gold = list(item.gold_tools)
        candidate_tools = [candidate.parent_tool for candidate in rerank_candidates_for_query]
        rows.append(
            {
                "row_id": f"{item.group}:{item.query_id}",
                "query_id": item.query_id,
                "query_text": item.query,
                "group": item.group,
                "gold_tools": gold,
                "gold_parent_tools": gold,
                "candidate_ids": [candidate.candidate_id for candidate in rerank_candidates_for_query],
                "candidate_tools": candidate_tools,
                "candidate_sources": candidate_sources,
                "docs_retrieval_pool_tools": [candidate.parent_tool for candidate in docs_ranked],
                "experience_retrieval_pool_tools": [candidate.parent_tool for candidate in exp_ranked],
                "predicted_tool": prompt.predicted_tool,
                "correct": prompt.predicted_tool in gold,
                "routed_correctly": prompt.predicted_tool in gold,
                "gold_in_top_k": any(tool in gold for tool in candidate_tools),
                "subset": subset_for(gold, heldout_tools),
                "candidate_distinct_count": len(set(candidate_tools)),
                "candidate_shortfall": len(set(candidate_tools)) < args.top_k,
                "parse_ok": prompt.parse_ok,
                "fallback_used": prompt.fallback_used,
                "prompt_hash": prompt.prompt_hash,
                "total_s": time.perf_counter() - started,
                "rerank_s": time.perf_counter() - rerank_started,
            }
        )
        if idx == 1 or idx % 50 == 0 or idx == len(queries):
            progress.log(
                "arm_progress",
                seed=seed,
                arm="two_index",
                completed_queries=idx,
                total_queries=len(queries),
                running_accuracy=sum(row["correct"] for row in rows) / len(rows),
                running_recall_at_5=sum(row["gold_in_top_k"] for row in rows) / len(rows),
                shortfalls=shortfalls,
            )
    if shortfalls:
        raise RuntimeError(f"E7 two-index seed {seed} had {shortfalls} candidate shortfalls under cap {args.retrieval_pool_size}")
    result = summarize_rows(rows)
    result.update(
        {
            "paper_eligible": True,
            "seed": seed,
            "arm": "two_index",
            "retriever": "jina",
            "rows": rows,
            "candidate_rule": "docs_plus_experience_backfill",
            "docs_budget": args.docs_budget,
            "candidate_shortfall_count": shortfalls,
            "mean_distinct_candidates": statistics.mean([row["candidate_distinct_count"] for row in rows]) if rows else 0.0,
        }
    )
    add_subsets(result)
    return result


def aggregate(paths: list[Path]) -> dict[str, Any]:
    cells = [json.loads(path.read_text()) for path in paths]
    out: dict[str, Any] = {"jina": {"two_index": {}}}
    for subset in ("heldout", "partial_heldout", "labeled"):
        vals = [float(cell["subset_metrics"][subset]["recall_at_5"]) for cell in cells]
        accs = [float(cell["subset_metrics"][subset]["accuracy"]) for cell in cells]
        out["jina"]["two_index"][subset] = {
            "mean_recall_at_5": statistics.mean(vals),
            "sd_recall_at_5": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "mean_accuracy": statistics.mean(accs),
            "sd_accuracy": statistics.stdev(accs) if len(accs) > 1 else 0.0,
            "n": cells[0]["subset_metrics"][subset]["n"] if cells else 0,
            "per_seed": {str(cell["seed"]): cell["subset_metrics"][subset] for cell in cells},
        }
    out["jina"]["two_index"]["overall_accuracy"] = statistics.mean([float(cell["accuracy"]) for cell in cells])
    out["jina"]["two_index"]["overall_recall_at_5"] = statistics.mean([float(cell["recall_at_5"]) for cell in cells])
    return out


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    commit, dirty = assert_clean_tree(allow_dirty=args.allow_dirty)
    progress = ProgressLogger(args.output_dir)
    progress.log("launch", repo_commit=commit, dirty_entry_count=len(dirty), output_dir=str(args.output_dir))

    input_summary = json.loads((args.input_dir / "summary.json").read_text())
    heldout_tools = json.loads((args.input_dir / "heldout_tools.json").read_text())
    heldout_set = set(heldout_tools)
    if sha256_texts(heldout_tools) != input_summary["heldout_tools_sha256"]:
        raise RuntimeError("E7 heldout_tools hash mismatch against input summary")

    embedder = resolve_local_embedder()
    jina = JinaAIClient(api_keys=[])
    written: list[Path] = []
    with temporary_env(
        {
            "JINA_LOCAL_EMBED_MODEL": embedder["model_path"],
            "JINA_LOCAL_EMBED_DEVICE": embedder["device"],
            "JINA_LOCAL_EMBED_LOCAL_ONLY": embedder["local_only"],
            "JINA_API_KEY": None,
        }
    ):
        groups = parse_csv(args.groups)
        queries_all = load_stabletoolbench_queries(args.stb_root, groups)
        train_items = load_toolbench_training_items(args.toolbench_instruction_dir)
        registry = build_tool_registry(queries_all + train_items)
        registry, _registry_filter = apply_junk_filter(registry)
        queries, _query_filter = filter_eval_queries(queries_all, registry)
        train_items, _train_filter = filter_experience_items(train_items, registry)
        train_items, heldout_filter = filter_heldout_pool(train_items, heldout_set)
        heldout_filter["pool_sha256"] = stable_hash([item_record(item) for item in train_items])
        if len(train_items) != int(input_summary["remaining_pool_count"]):
            raise RuntimeError(f"E7 pool count mismatch: {len(train_items)} != {input_summary['remaining_pool_count']}")
        if heldout_filter["remaining_items_with_heldout_label"] != 0:
            raise RuntimeError("E7 held-out label filter failed")

        backend = load_local_backend(args.model_path)
        docs_pkg, docs_hash = build_doc_package(registry)
        progress.log("docs_package_done", artifact_count=len(docs_pkg.artifacts), package_sha256=docs_hash)
        for seed in parse_ints(args.seeds):
            clients = limit_client_items(assign_clients(train_items, args.client_count, args.partition_mode, seed), args.max_items_per_client, seed)
            client_packages = {}
            client_hashes = {}
            for client_id, items in sorted(clients.items()):
                pkg, pkg_hash = build_client_package(client_id, items, registry, jina, args.embed_model)
                client_packages[client_id] = pkg
                client_hashes[client_id] = pkg_hash
                progress.log("client_package_done", seed=seed, client_id=client_id, item_count=len(items), artifact_count=len(pkg.artifacts), package_sha256=pkg_hash)
            exp_pkg, exp_hash = combine_packages("concat_experience_only", [client_packages[k] for k in sorted(client_packages)])
            concat_pkg, concat_hash = combine_packages("concat_with_docs", [docs_pkg, *[client_packages[k] for k in sorted(client_packages)]])
            progress.log("arm_begin", seed=seed, arm="two_index", docs_sha256=docs_hash, experience_sha256=exp_hash, rerank_sha256=concat_hash)
            result = evaluate_two_index(
                seed=seed,
                queries=queries,
                heldout_tools=heldout_set,
                docs_pkg=docs_pkg,
                exp_pkg=exp_pkg,
                rerank_pkg=concat_pkg,
                jina=jina,
                backend=backend,
                args=args,
                progress=progress,
            )
            result.update(
                {
                    "repo_commit": commit,
                    "input_dir": str(args.input_dir),
                    "input_repo_commit": input_summary.get("repo_commit"),
                    "data_mode": "toolbench_train_category_holdout",
                    "heldout_categories_sha256": input_summary["heldout_categories_sha256"],
                    "heldout_tools_sha256": input_summary["heldout_tools_sha256"],
                    "heldout_tool_count": len(heldout_tools),
                    "pool_sha256": heldout_filter["pool_sha256"],
                    "heldout_filter": heldout_filter,
                    "client_sha256": client_hashes,
                    "tool_doc_sha256": docs_hash,
                    "experience_sha256": exp_hash,
                    "compendium": {"global_sha256": concat_hash, "artifact_count": len(concat_pkg.artifacts)},
                    "metric_definition": {"correct": "predicted_tool in gold_tools", "recall_at_5": "any gold tool in five distinct candidate tools"},
                    "config": {
                        "retrieval_pool_size": args.retrieval_pool_size,
                        "candidate_rule": "docs_plus_experience_backfill",
                        "docs_budget": args.docs_budget,
                        "top_k": args.top_k,
                        "reranked": True,
                        "reranker_variant": args.reranker_variant,
                    },
                }
            )
            out_path = args.output_dir / f"seed_{seed}" / "jina" / "two_index.json"
            save_json(out_path, result)
            written.append(out_path)
            progress.log(
                "arm_done",
                seed=seed,
                arm="two_index",
                accuracy=result["accuracy"],
                recall_at_5=result["recall_at_5"],
                heldout_recall_at_5=result["subset_metrics"]["heldout"]["recall_at_5"],
                labeled_recall_at_5=result["subset_metrics"]["labeled"]["recall_at_5"],
            )

    summary = {
        "paper_eligible": True,
        "repo_commit": commit,
        "input_summary": str(args.input_dir / "summary.json"),
        "input_repo_commit": input_summary.get("repo_commit"),
        "seeds": parse_ints(args.seeds),
        "retrievers": ["jina"],
        "arms": ["two_index"],
        "candidate_rule": "docs_plus_experience_backfill",
        "aggregate": aggregate(written),
        "result_files": [str(path.relative_to(CANONICAL_ROOT)) for path in written],
    }
    save_json(args.output_dir / "summary.json", summary)
    progress.log("complete", output_dir=str(args.output_dir), result_count=len(written))
    print(json.dumps(summary["aggregate"], indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
