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
from scripts.run_reranker_prompt_sweep import RoutedCandidate, render_precautions, run_prompt
from scripts.run_stabletoolbench_federated import (
    DEFAULT_MODEL_PATH,
    DEFAULT_STB_ROOT,
    DEFAULT_TOOLBENCH_INSTRUCTION_DIR,
    GROUPS,
    ProgressLogger,
    apply_junk_filter,
    assert_clean_tree,
    assign_clients,
    build_client_package,
    build_doc_package,
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
    short_text,
    stable_hash,
    summarize_rows,
    temporary_env,
)
from scripts.run_stabletoolbench_heldout import filter_heldout_pool, sha256_texts
from scripts.run_stabletoolbench_heldout_retriever_compare import (
    BM25Index,
    DEFAULT_BGE,
    bge_encode,
    build_bge_encoder,
    retrieve,
    sha256_file,
)


DEFAULT_INPUT_DIR = CANONICAL_ROOT / "artifacts" / "results" / "stabletoolbench_category_holdout_e7_inputs_r1"
DEFAULT_OUTPUT_DIR = CANONICAL_ROOT / "artifacts" / "results" / "stabletoolbench_category_holdout_e7_concat_r1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E7 category hold-out with CONCAT pooled-experience arm.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stb-root", type=Path, default=DEFAULT_STB_ROOT)
    parser.add_argument("--toolbench-instruction-dir", type=Path, default=DEFAULT_TOOLBENCH_INSTRUCTION_DIR)
    parser.add_argument("--groups", default=",".join(GROUPS))
    parser.add_argument("--seeds", default="42,123,456")
    parser.add_argument("--client-count", type=int, default=5)
    parser.add_argument("--max-items-per-client", type=int, default=5000)
    parser.add_argument("--partition-mode", choices=("category", "iid"), default="category")
    parser.add_argument("--retrievers", default="jina,bm25,bge")
    parser.add_argument("--rerank-retrievers", default="jina")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retrieval-pool-size", type=int, default=200)
    parser.add_argument("--retrieval-mode", default="distinct_tool_topk")
    parser.add_argument("--reranker-variant", default="V3")
    parser.add_argument("--embed-model", default="jina-embeddings-v2-base-en")
    parser.add_argument("--bge-model-path", type=Path, default=DEFAULT_BGE)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--no-refuse-shortfall", action="store_true")
    return parser.parse_args()


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]




def package_to_candidates_no_embed(package) -> list[RoutedCandidate]:
    candidates: list[RoutedCandidate] = []
    for artifact in package.artifacts:
        payload = artifact.structured_payload or {}
        when_to_use: list[str] = []
        if isinstance(payload.get("tool_description"), str) and payload["tool_description"].strip():
            when_to_use.append(payload["tool_description"].strip())
        if isinstance(payload.get("scenario_context"), str) and payload["scenario_context"].strip():
            when_to_use.append(payload["scenario_context"].strip())
        if isinstance(payload.get("annex_summary"), str) and payload["annex_summary"].strip():
            when_to_use.append(payload["annex_summary"].strip())
        if not when_to_use:
            when_to_use = [short_text(artifact.text, 200)]
        precautions = [str(item).strip() for item in payload.get("precautions", []) if str(item).strip()] if isinstance(payload.get("precautions"), list) else []
        conflict_log = [str(item).strip() for item in payload.get("conflict_log", []) if str(item).strip()] if isinstance(payload.get("conflict_log"), list) else []
        candidates.append(
            RoutedCandidate(
                candidate_id=artifact.signature,
                label=str(artifact.metadata.get("tool") or artifact.signature),
                parent_tool=str(artifact.metadata.get("tool") or artifact.signature),
                when_to_use=when_to_use,
                do_not_use_when=render_precautions(precautions, conflict_log),
                text=artifact.text,
                embedding=[],
                provenance=str(artifact.metadata.get("artifact_origin") or artifact.metadata.get("source_group") or artifact.metadata.get("category") or "stabletoolbench"),
            )
        )
    return candidates


def prepare_candidates_fast(package, retriever: str, jina: JinaAIClient, embed_model: str, bge_model: Any | None):
    if retriever == "jina":
        candidates = package_to_candidates(package, jina, embed_model)
        vectors = np.asarray([candidate.embedding for candidate in candidates], dtype=np.float32)
        if vectors.size:
            norms = np.linalg.norm(vectors, axis=1)
            norms[norms == 0.0] = 1.0
            vectors = vectors / norms[:, None]
        return candidates, vectors, {"retriever": "jina", "embed_model": embed_model}
    candidates = package_to_candidates_no_embed(package)
    texts = [candidate.text for candidate in candidates]
    if retriever == "bm25":
        index = BM25Index(texts)
        return candidates, index, {"retriever": "bm25", "tokenizer": "regex:[A-Za-z0-9_]+", "k1": index.k1, "b": index.b}
    if retriever == "bge":
        if bge_model is None:
            raise RuntimeError("BGE model requested but not loaded")
        return candidates, bge_encode(bge_model, texts), {"retriever": "bge", "model_path": str(args.bge_model_path) if False else "bge-base-en-v1.5"}
    raise ValueError(f"unknown retriever: {retriever}")


def subset_for(gold: list[str], heldout: set[str]) -> str:
    if gold and all(tool in heldout for tool in gold):
        return "heldout"
    if gold and any(tool in heldout for tool in gold):
        return "partial_heldout"
    return "labeled"


def item_record(item: Any) -> dict[str, Any]:
    return {"query_id": item.query_id, "query": item.query, "gold_tools": item.gold_tools}


def subset_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "accuracy": sum(1 for row in rows if row["correct"]) / len(rows) if rows else 0.0,
        "recall_at_5": sum(1 for row in rows if row["gold_in_top_k"]) / len(rows) if rows else 0.0,
        "retrieval_top1": sum(1 for row in rows if (row.get("candidate_tools") or [""])[0] in set(row.get("gold_tools") or [])) / len(rows) if rows else 0.0,
    }


def add_subsets(result: dict[str, Any]) -> None:
    result["subset_metrics"] = {}
    for subset in ("heldout", "partial_heldout", "labeled"):
        result["subset_metrics"][subset] = subset_metrics([row for row in result["rows"] if row.get("subset") == subset])


def evaluate_arm(
    *,
    arm: str,
    retriever: str,
    package,
    queries: list[Any],
    heldout_tools: set[str],
    jina: JinaAIClient,
    bge_model: Any | None,
    backend: Any | None,
    args: argparse.Namespace,
    progress: ProgressLogger,
    seed: int,
) -> dict[str, Any]:
    candidates, index, retriever_meta = prepare_candidates_fast(package, retriever, jina, args.embed_model, bge_model)
    query_embeddings = (
        __import__("scripts.run_stabletoolbench_federated", fromlist=["batched_query_embeddings"]).batched_query_embeddings(
            jina, [item.query for item in queries], args.embed_model
        )
        if retriever == "jina"
        else [None] * len(queries)
    )
    rerank = retriever in set(parse_csv(args.rerank_retrievers))
    if rerank and backend is None:
        raise RuntimeError(f"reranking requested for {retriever} but no backend is loaded")
    rows: list[dict[str, Any]] = []
    shortfalls = 0
    for idx, (item, query_embedding) in enumerate(zip(queries, query_embeddings), start=1):
        started = time.perf_counter()
        ranked, pool_tools = retrieve(candidates, index, retriever, item.query, query_embedding, bge_model, args)
        candidate_tools = [candidate.parent_tool for candidate in ranked]
        distinct_count = len(set(candidate_tools))
        if distinct_count < args.top_k:
            shortfalls += 1
        if rerank and ranked:
            maybe_cuda_synchronize(backend)
            rerank_started = time.perf_counter()
            prompt = run_prompt(backend, args.reranker_variant, "toolbench", item.query, ranked, [], ranked[0])
            maybe_cuda_synchronize(backend)
            predicted = prompt.predicted_tool
            parse_ok = prompt.parse_ok
            fallback = prompt.fallback_used
            prompt_hash = prompt.prompt_hash
            rerank_s = time.perf_counter() - rerank_started
        else:
            predicted = ranked[0].parent_tool if ranked else ""
            parse_ok = True
            fallback = False
            prompt_hash = ""
            rerank_s = 0.0
        gold = list(item.gold_tools)
        rows.append(
            {
                "row_id": f"{item.group}:{item.query_id}",
                "query_id": item.query_id,
                "query_text": item.query,
                "group": item.group,
                "gold_tools": gold,
                "gold_parent_tools": gold,
                "candidate_ids": [candidate.candidate_id for candidate in ranked],
                "candidate_tools": candidate_tools,
                "retrieval_pool_tools": pool_tools,
                "predicted_tool": predicted,
                "correct": predicted in gold,
                "routed_correctly": predicted in gold,
                "gold_in_top_k": any(tool in gold for tool in candidate_tools),
                "subset": subset_for(gold, heldout_tools),
                "candidate_distinct_count": distinct_count,
                "candidate_shortfall": distinct_count < args.top_k,
                "parse_ok": parse_ok,
                "fallback_used": fallback,
                "prompt_hash": prompt_hash,
                "total_s": time.perf_counter() - started,
                "rerank_s": rerank_s,
            }
        )
        if idx == 1 or idx % 50 == 0 or idx == len(queries):
            progress.log(
                "arm_progress",
                seed=seed,
                retriever=retriever,
                arm=arm,
                completed_queries=idx,
                total_queries=len(queries),
                running_accuracy=sum(row["correct"] for row in rows) / len(rows),
                running_recall_at_5=sum(row["gold_in_top_k"] for row in rows) / len(rows),
                shortfalls=shortfalls,
            )
    if shortfalls and not args.no_refuse_shortfall:
        raise RuntimeError(f"E7 {seed}/{retriever}/{arm} had {shortfalls} candidate shortfalls under cap {args.retrieval_pool_size}")
    result = summarize_rows(rows)
    result.update(
        {
            "paper_eligible": True,
            "seed": seed,
            "arm": arm,
            "retriever": retriever,
            "rows": rows,
            "retriever_metadata": retriever_meta,
            "candidate_rule": "distinct5_walkdown",
            "candidate_shortfall_count": shortfalls,
            "mean_distinct_candidates": statistics.mean([row["candidate_distinct_count"] for row in rows]) if rows else 0.0,
        }
    )
    add_subsets(result)
    return result


def aggregate(paths: list[Path]) -> dict[str, Any]:
    cells = [json.loads(path.read_text()) for path in paths]
    out: dict[str, Any] = {}
    for retriever in sorted({cell["retriever"] for cell in cells}):
        out[retriever] = {}
        for arm in sorted({cell["arm"] for cell in cells if cell["retriever"] == retriever}):
            group = [cell for cell in cells if cell["retriever"] == retriever and cell["arm"] == arm]
            out[retriever][arm] = {}
            for subset in ("heldout", "partial_heldout", "labeled"):
                vals = [float(cell["subset_metrics"][subset]["recall_at_5"]) for cell in group]
                accs = [float(cell["subset_metrics"][subset]["accuracy"]) for cell in group]
                out[retriever][arm][subset] = {
                    "mean_recall_at_5": statistics.mean(vals) if vals else 0.0,
                    "sd_recall_at_5": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                    "mean_accuracy": statistics.mean(accs) if accs else 0.0,
                    "sd_accuracy": statistics.stdev(accs) if len(accs) > 1 else 0.0,
                    "per_seed": {str(cell["seed"]): cell["subset_metrics"][subset] for cell in group},
                }
            out[retriever][arm]["overall_accuracy"] = statistics.mean([float(cell["accuracy"]) for cell in group])
            out[retriever][arm]["overall_recall_at_5"] = statistics.mean([float(cell["recall_at_5"]) for cell in group])
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

    groups = parse_csv(args.groups)
    seeds = parse_ints(args.seeds)
    retrievers = parse_csv(args.retrievers)
    rerank_retrievers = set(parse_csv(args.rerank_retrievers))

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
        queries_all = load_stabletoolbench_queries(args.stb_root, groups)
        train_items = load_toolbench_training_items(args.toolbench_instruction_dir)
        registry = build_tool_registry(queries_all + train_items)
        registry, registry_filter = apply_junk_filter(registry)
        queries, query_filter = filter_eval_queries(queries_all, registry)
        train_items, train_filter = filter_experience_items(train_items, registry)
        train_items, heldout_filter = filter_heldout_pool(train_items, heldout_set)
        heldout_filter["pool_sha256"] = stable_hash([item_record(item) for item in train_items])
        if len(train_items) != int(input_summary["remaining_pool_count"]):
            raise RuntimeError(f"E7 pool count mismatch: {len(train_items)} != {input_summary['remaining_pool_count']}")
        if heldout_filter["remaining_items_with_heldout_label"] != 0:
            raise RuntimeError("E7 held-out label filter failed")
        progress.log(
            "data_ready",
            query_count=len(queries),
            train_count=len(train_items),
            heldout_tool_count=len(heldout_tools),
            heldout_query_count=input_summary["eval_summary"]["heldout_query_count"],
            registry_filter=registry_filter,
            query_filter=query_filter,
            train_filter=train_filter,
            heldout_filter=heldout_filter,
        )

        backend = load_local_backend(args.model_path) if rerank_retrievers else None
        bge_model = None
        if "bge" in retrievers:
            device = os.environ.get("BGE_DEVICE") or ("cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
            progress.log("load_bge_begin", model_path=str(args.bge_model_path), device=device)
            bge_model = build_bge_encoder(args.bge_model_path, device=device)
            progress.log("load_bge_done", model_sha256=sha256_file(args.bge_model_path / "config.json"))

        docs_pkg, docs_hash = build_doc_package(registry)
        progress.log("docs_package_done", artifact_count=len(docs_pkg.artifacts), package_sha256=docs_hash)
        for seed in seeds:
            clients = limit_client_items(assign_clients(train_items, args.client_count, args.partition_mode, seed), args.max_items_per_client, seed)
            client_packages = {}
            client_hashes = {}
            for client_id, items in sorted(clients.items()):
                pkg, pkg_hash = build_client_package(client_id, items, registry, jina, args.embed_model)
                client_packages[client_id] = pkg
                client_hashes[client_id] = pkg_hash
                progress.log("client_package_done", seed=seed, client_id=client_id, item_count=len(items), artifact_count=len(pkg.artifacts), package_sha256=pkg_hash)
            concat_pkg, concat_hash = combine_packages("concat_with_docs", [docs_pkg, *[client_packages[k] for k in sorted(client_packages)]])
            packages = {"docs_only": (docs_pkg, docs_hash), "concat": (concat_pkg, concat_hash)}
            for retriever in retrievers:
                for arm, (package, package_hash) in packages.items():
                    progress.log("arm_begin", seed=seed, retriever=retriever, arm=arm, artifact_count=len(package.artifacts), package_sha256=package_hash)
                    result = evaluate_arm(
                        arm=arm,
                        retriever=retriever,
                        package=package,
                        queries=queries,
                        heldout_tools=heldout_set,
                        jina=jina,
                        bge_model=bge_model,
                        backend=backend,
                        args=args,
                        progress=progress,
                        seed=seed,
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
                            "compendium": {"global_sha256": package_hash, "artifact_count": len(package.artifacts)},
                            "tool_doc_sha256": docs_hash,
                            "merge_mode": "server_concatenate_no_dedup_no_field_merge" if arm == "concat" else "docs_only",
                            "metric_definition": {"correct": "predicted_tool in gold_tools", "recall_at_5": "any gold tool in five distinct candidate tools"},
                            "config": {
                                "retrieval_pool_size": args.retrieval_pool_size,
                                "retrieval_mode": args.retrieval_mode,
                                "top_k": args.top_k,
                                "reranked": retriever in rerank_retrievers,
                                "reranker_variant": args.reranker_variant if retriever in rerank_retrievers else None,
                            },
                        }
                    )
                    out_path = args.output_dir / f"seed_{seed}" / retriever / f"{arm}.json"
                    save_json(out_path, result)
                    written.append(out_path)
                    progress.log(
                        "arm_done",
                        seed=seed,
                        retriever=retriever,
                        arm=arm,
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
        "seeds": seeds,
        "retrievers": retrievers,
        "arms": ["docs_only", "concat"],
        "rerank_retrievers": sorted(rerank_retrievers),
        "candidate_rule": "distinct5_walkdown",
        "candidate_shortfall_refusal": not args.no_refuse_shortfall,
        "e7_arm_amendment": "CONCAT is the pooled-experience arm; equivalence to SYNAPSE is established in Section 7.",
        "aggregate": aggregate(written),
        "result_files": [str(path.relative_to(CANONICAL_ROOT)) for path in written],
    }
    suffix = "_".join(str(seed) for seed in seeds)
    save_json(args.output_dir / f"summary_seeds_{suffix}.json", summary)
    if len(seeds) == 3:
        save_json(args.output_dir / "summary.json", summary)
    progress.log("complete", output_dir=str(args.output_dir), result_count=len(written))
    print(json.dumps(summary["aggregate"], indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
