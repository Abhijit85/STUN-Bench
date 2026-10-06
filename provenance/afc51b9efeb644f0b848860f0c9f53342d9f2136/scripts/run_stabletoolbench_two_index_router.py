#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
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
    GROUPS,
    ProgressLogger,
    apply_junk_filter,
    assert_clean_tree,
    batched_query_embeddings,
    build_ranked_candidates,
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
from scripts.run_stabletoolbench_heldout_retriever_compare import BM25Index, bge_encode, build_bge_encoder, mcnemar_recall, sha256_file
from synapse.knowledge.compendium import KnowledgeArtifact, KnowledgePackage

DEFAULT_BGE = Path("<HF_CACHE>/models--BAAI--bge-base-en-v1.5/snapshots/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E8: two-index held-out router with fused description and experience candidates.")
    parser.add_argument("--stb-root", type=Path, default=DEFAULT_STB_ROOT)
    parser.add_argument("--groups", type=str, default=",".join(GROUPS))
    parser.add_argument("--seeds", type=str, default="42,123,456")
    parser.add_argument("--e6-dirs", type=str, required=True, help="Comma-separated seed=dir entries with E6 packages.")
    parser.add_argument("--retrievers", type=str, default="jina,bm25,bge")
    parser.add_argument("--rules", type=str, default="union_rrf,union_interleave,docs_plus_experience_backfill")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retrieval-pool-size", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--reranker-variant", type=str, default="V3")
    parser.add_argument("--embed-model", type=str, default="jina-embeddings-v2-base-en")
    parser.add_argument("--bge-model-path", type=Path, default=DEFAULT_BGE)
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=CANONICAL_ROOT / "artifacts" / "results" / "stabletoolbench_two_index_router_e8_r1")
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args()


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_seed_dirs(value: str) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for part in parse_csv(value):
        seed, path = part.split("=", 1)
        out[int(seed)] = Path(path)
    return out


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalize_matrix(vectors: list[list[float]]) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.size:
        norms = np.linalg.norm(matrix, axis=1)
        norms[norms == 0.0] = 1.0
        matrix = matrix / norms[:, None]
    return matrix


def experience_only_package(package: KnowledgePackage) -> KnowledgePackage:
    artifacts = [
        artifact
        for artifact in package.artifacts
        if artifact.metadata.get("artifact_origin") != "tool_doc"
        and artifact.structured_payload
        and artifact.structured_payload.get("type") != "tool_doc"
    ]
    return KnowledgePackage(
        source_id=f"{package.source_id}_experience_only",
        artifacts=[
            KnowledgeArtifact(
                signature=artifact.signature,
                text=artifact.text,
                structured_payload=artifact.structured_payload,
                metadata=dict(artifact.metadata),
            )
            for artifact in artifacts
        ],
        metadata={**package.metadata, "artifact_filter": "exclude_tool_doc"},
    )


def prepare_index(package: KnowledgePackage, retriever: str, jina_client: JinaAIClient, embed_model: str, bge_model: Any | None):
    candidates = package_to_candidates(package, jina_client, embed_model)
    texts = [candidate.text for candidate in candidates]
    if retriever == "jina":
        index = normalize_matrix([candidate.embedding for candidate in candidates]) if candidates else np.zeros((0, 0), dtype=np.float32)
        meta = {"retriever": "jina", "embed_model": embed_model}
    elif retriever == "bm25":
        index = BM25Index(texts)
        meta = {"retriever": "bm25", "tokenizer": "regex:[A-Za-z0-9_]+", "k1": index.k1, "b": index.b}
    elif retriever == "bge":
        if bge_model is None:
            raise RuntimeError("BGE model requested but not loaded")
        index = bge_encode(bge_model, texts)
        meta = {"retriever": "bge", "model_path": str(getattr(bge_model, "_model_card_text", "") or "")}
    else:
        raise ValueError(f"unknown retriever: {retriever}")
    return candidates, index, meta


def score_query(index: Any, retriever: str, query: str, jina_query_embedding: list[float] | None, bge_model: Any | None) -> np.ndarray:
    if retriever == "bm25":
        return index.encode_query(query)
    if retriever == "bge":
        q = bge_encode(bge_model, [query])[0]
        return index @ q if getattr(index, "size", 0) else np.asarray([], dtype=np.float32)
    q = np.asarray(jina_query_embedding, dtype=np.float32)
    norm = float(np.linalg.norm(q))
    if norm > 0.0:
        q = q / norm
    return index @ q if getattr(index, "size", 0) else np.asarray([], dtype=np.float32)


def top_distinct(candidates: list[Any], scores: np.ndarray, pool_size: int) -> list[Any]:
    ranked, _ = build_ranked_candidates(candidates, scores, pool_size, pool_size, "distinct_tool_topk")
    return ranked


def fuse_candidates(rule: str, docs: list[Any], exp: list[Any], top_k: int, rrf_k: int) -> tuple[list[Any], list[dict[str, Any]]]:
    by_tool: dict[str, Any] = {}
    sources: dict[str, set[str]] = {}
    if rule == "union_rrf":
        scores: dict[str, float] = {}
        for source, ranked in (("docs", docs), ("experience", exp)):
            for rank, candidate in enumerate(ranked, start=1):
                tool = candidate.parent_tool
                by_tool.setdefault(tool, candidate)
                sources.setdefault(tool, set()).add(source)
                scores[tool] = scores.get(tool, 0.0) + 1.0 / (rrf_k + rank)
        ordered = sorted(scores, key=lambda tool: (-scores[tool], tool))[:top_k]
    elif rule == "union_interleave":
        ordered = []
        for idx in range(max(len(docs), len(exp))):
            for source, ranked in (("docs", docs), ("experience", exp)):
                if idx >= len(ranked):
                    continue
                tool = ranked[idx].parent_tool
                by_tool.setdefault(tool, ranked[idx])
                sources.setdefault(tool, set()).add(source)
                if tool not in ordered:
                    ordered.append(tool)
                if len(ordered) == top_k:
                    break
            if len(ordered) == top_k:
                break
    elif rule == "docs_plus_experience_backfill":
        ordered = []
        for source, ranked, limit in (("docs", docs, 3), ("experience", exp, top_k)):
            added = 0
            for candidate in ranked:
                tool = candidate.parent_tool
                by_tool.setdefault(tool, candidate)
                sources.setdefault(tool, set()).add(source)
                if tool not in ordered:
                    ordered.append(tool)
                    added += 1
                if len(ordered) == top_k or added == limit:
                    break
            if len(ordered) == top_k:
                break
    else:
        raise ValueError(f"unknown fusion rule: {rule}")
    fused = [by_tool[tool] for tool in ordered[:top_k]]
    source_rows = [{"tool": tool, "sources": sorted(sources.get(tool, []))} for tool in ordered[:top_k]]
    return fused, source_rows


def summarize_subset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "accuracy": sum(1 for row in rows if row["correct"]) / len(rows) if rows else 0.0,
        "recall_at_5": sum(1 for row in rows if row["gold_in_top_k"]) / len(rows) if rows else 0.0,
        "retrieval_top1": sum(1 for row in rows if (row.get("candidate_tools") or [""])[0] in set(row.get("gold_tools") or [])) / len(rows) if rows else 0.0,
    }


def add_subsets(result: dict[str, Any]) -> None:
    result["subset_metrics"] = {}
    for subset in ("heldout", "labeled"):
        result["subset_metrics"][subset] = summarize_subset([row for row in result["rows"] if row["subset"] == subset])


def evaluate_rule(*, retriever: str, rule: str, docs_idx: tuple[Any, Any], exp_idx: tuple[Any, Any], rerank_by_tool: dict[str, Any], test_items: list[Any], heldout_tools: set[str], jina_embeddings: list[list[float] | None], bge_model: Any | None, backend: Any, args: argparse.Namespace, progress: ProgressLogger, seed: int) -> dict[str, Any]:
    docs_candidates, docs_index = docs_idx
    exp_candidates, exp_index = exp_idx
    rows = []
    for idx, (item, jina_embedding) in enumerate(zip(test_items, jina_embeddings), start=1):
        started = time.perf_counter()
        docs_top = top_distinct(docs_candidates, score_query(docs_index, retriever, item.query, jina_embedding, bge_model), args.retrieval_pool_size)
        exp_top = top_distinct(exp_candidates, score_query(exp_index, retriever, item.query, jina_embedding, bge_model), args.retrieval_pool_size)
        fused, candidate_sources = fuse_candidates(rule, docs_top, exp_top, args.top_k, args.rrf_k)
        rerank_candidates = [rerank_by_tool[candidate.parent_tool] for candidate in fused if candidate.parent_tool in rerank_by_tool]
        top_candidate = rerank_candidates[0] if rerank_candidates else None
        if top_candidate is None:
            predicted = ""
            parse_ok = False
            fallback = True
            prompt_hash = ""
            rerank_s = 0.0
        else:
            maybe_cuda_synchronize(backend)
            rerank_started = time.perf_counter()
            result = run_prompt(backend, args.reranker_variant, "toolbench", item.query, rerank_candidates, [], top_candidate)
            maybe_cuda_synchronize(backend)
            rerank_s = time.perf_counter() - rerank_started
            predicted = result.predicted_tool
            parse_ok = result.parse_ok
            fallback = result.fallback_used
            prompt_hash = result.prompt_hash
        gold = item.gold_tools
        retrieved_tools = [candidate.parent_tool for candidate in fused]
        rows.append(
            {
                "query_id": item.query_id,
                "query_text": item.query,
                "group": item.group,
                "gold_tools": gold,
                "candidate_tools": retrieved_tools,
                "candidate_ids": [candidate.candidate_id for candidate in fused],
                "candidate_sources": candidate_sources,
                "predicted_tool": predicted,
                "correct": predicted in gold,
                "routed_correctly": predicted in gold,
                "gold_in_top_k": any(tool in gold for tool in retrieved_tools),
                "subset": "heldout" if gold and all(tool in heldout_tools for tool in gold) else "labeled",
                "oracle_retrieval": False,
                "parse_ok": parse_ok,
                "fallback_used": fallback,
                "rerank_s": rerank_s,
                "total_s": time.perf_counter() - started,
                "prompt_hash": prompt_hash,
            }
        )
        if idx == 1 or idx % 25 == 0 or idx == len(test_items):
            progress.log("arm_progress", seed=seed, retriever=retriever, rule=rule, completed_queries=idx, total_queries=len(test_items), running_accuracy=sum(row["correct"] for row in rows) / len(rows), running_recall_at_5=sum(row["gold_in_top_k"] for row in rows) / len(rows))
    result = summarize_rows(rows)
    result.update({"seed": seed, "retriever": retriever, "arm": rule, "fusion_rule": rule, "rows": rows})
    add_subsets(result)
    return result


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for retriever in sorted({r["retriever"] for r in results}):
        out[retriever] = {}
        for arm in sorted({r["arm"] for r in results if r["retriever"] == retriever}):
            group = [r for r in results if r["retriever"] == retriever and r["arm"] == arm]
            out[retriever][arm] = {}
            for subset in ("heldout", "labeled"):
                acc = [r["subset_metrics"][subset]["accuracy"] for r in group]
                rec = [r["subset_metrics"][subset]["recall_at_5"] for r in group]
                ns = [r["subset_metrics"][subset]["n"] for r in group]
                out[retriever][arm][subset] = {
                    "mean_n": statistics.mean(ns) if ns else 0,
                    "mean_accuracy": statistics.mean(acc) if acc else 0.0,
                    "sd_accuracy": statistics.stdev(acc) if len(acc) > 1 else 0.0,
                    "mean_recall_at_5": statistics.mean(rec) if rec else 0.0,
                    "sd_recall_at_5": statistics.stdev(rec) if len(rec) > 1 else 0.0,
                }
    return out


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    commit, dirty = assert_clean_tree(allow_dirty=args.allow_dirty)
    progress = ProgressLogger(args.output_dir)
    progress.log("launch", repo_commit=commit, dirty_entry_count=len(dirty))
    seed_dirs = parse_seed_dirs(args.e6_dirs)
    seeds = [int(seed) for seed in parse_csv(args.seeds)]
    retrievers = parse_csv(args.retrievers)
    rules = parse_csv(args.rules)
    groups = parse_csv(args.groups)
    embedder_info = resolve_local_embedder()
    jina_client = JinaAIClient(api_keys=[])
    with temporary_env({"JINA_LOCAL_EMBED_MODEL": embedder_info["model_path"], "JINA_LOCAL_EMBED_DEVICE": embedder_info["device"], "JINA_LOCAL_EMBED_LOCAL_ONLY": embedder_info["local_only"], "JINA_API_KEY": None}):
        all_queries = load_stabletoolbench_queries(args.stb_root, groups)
        registry = build_tool_registry(all_queries)
        registry, junk_info = apply_junk_filter(registry)
        queries, eval_filter = filter_eval_queries(all_queries, registry)
        heldout_tools, heldout_sanity = heldout_tool_set(queries, HELDOUT_GROUPS)
        heldout_set = set(heldout_tools)
        heldout_sha = stable_hash(heldout_tools)
        progress.log("data_ready", test_count=len(queries), heldout_tool_count=len(heldout_tools), heldout_tools_sha256=heldout_sha, junk_filter=junk_info, eval_filter=eval_filter)

        backend = load_local_backend(args.model_path)
        bge_model = None
        if "bge" in retrievers:
            bge_model = build_bge_encoder(args.bge_model_path, os.environ.get("BGE_DEVICE") or ("cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"))
            progress.log("load_bge_done", model_path=str(args.bge_model_path), model_sha256=sha256_file(args.bge_model_path / "config.json"))

        all_results = []
        for seed in seeds:
            e6_dir = seed_dirs[seed]
            package_dir = e6_dir / f"seed_{seed}" / "packages"
            flat_package, flat_hash = load_package_file(package_dir / "flat_pool.json")
            synapse_package, synapse_hash = load_package_file(package_dir / "synapse_shared.json")
            experience_package = experience_only_package(synapse_package)
            experience_hash = stable_hash(
                [{"signature": a.signature, "text": a.text, "metadata": a.metadata, "structured_payload": a.structured_payload} for a in experience_package.artifacts]
            )
            rerank_by_tool = {}
            for candidate in package_to_candidates(synapse_package, jina_client, args.embed_model):
                rerank_by_tool.setdefault(candidate.parent_tool, candidate)
            progress.log("seed_ready", seed=seed, flat_pool_sha256=flat_hash, synapse_sha256=synapse_hash, experience_sha256=experience_hash, docs_doc_count=len(flat_package.artifacts), experience_doc_count=len(experience_package.artifacts), heldout_tools_sha256=heldout_sha)
            seed_results = []
            for retriever in retrievers:
                bge_for_retriever = bge_model if retriever == "bge" else None
                docs_candidates, docs_index, docs_meta = prepare_index(flat_package, retriever, jina_client, args.embed_model, bge_for_retriever)
                exp_candidates, exp_index, exp_meta = prepare_index(experience_package, retriever, jina_client, args.embed_model, bge_for_retriever)
                jina_embeddings = batched_query_embeddings(jina_client, [item.query for item in queries], args.embed_model) if retriever == "jina" else [None] * len(queries)
                retriever_rules = rules if retriever == "jina" else [rule for rule in rules if rule == "union_rrf"]
                for rule in retriever_rules:
                    progress.log("arm_begin", seed=seed, retriever=retriever, rule=rule)
                    result = evaluate_rule(retriever=retriever, rule=rule, docs_idx=(docs_candidates, docs_index), exp_idx=(exp_candidates, exp_index), rerank_by_tool=rerank_by_tool, test_items=queries, heldout_tools=heldout_set, jina_embeddings=jina_embeddings, bge_model=bge_for_retriever, backend=backend, args=args, progress=progress, seed=seed)
                    result.update({
                        "paper_eligible": True,
                        "repo_commit": commit,
                        "data_mode": "toolbench_train",
                        "heldout_tools_sha256": heldout_sha,
                        "heldout_sanity": heldout_sanity,
                        "source": {
                            "e6_dir": str(e6_dir),
                            "flat_pool_sha256": flat_hash,
                            "synapse_sha256": synapse_hash,
                            "experience_sha256": experience_hash,
                        },
                        "index_document_counts": {"docs": len(flat_package.artifacts), "experience": len(experience_package.artifacts)},
                        "retriever_metadata": {"docs": docs_meta, "experience": exp_meta},
                        "metric_definition": {"correct": "predicted_tool in gold_tools", "recall_at_5": "any gold tool appears among fused candidate tools"},
                    })
                    save_json(args.output_dir / f"seed_{seed}" / retriever / f"{rule}.json", result)
                    seed_results.append(result)
                    all_results.append(result)
                    progress.log("arm_done", seed=seed, retriever=retriever, rule=rule, accuracy=result["accuracy"], recall_at_5=result["recall_at_5"], heldout_recall_at_5=result["subset_metrics"]["heldout"]["recall_at_5"], labeled_recall_at_5=result["subset_metrics"]["labeled"]["recall_at_5"])
            save_json(args.output_dir / f"seed_{seed}" / "seed_summary.json", {"paper_eligible": True, "seed": seed, "aggregate": aggregate(seed_results)})
            progress.log("seed_done", seed=seed)
        summary = {
            "paper_eligible": True,
            "config": {
                "repo_commit": commit,
                "seeds": seeds,
                "retrievers": retrievers,
                "rules": rules,
                "heldout_tools_sha256": heldout_sha,
                "top_k": args.top_k,
                "retrieval_pool_size": args.retrieval_pool_size,
                "rrf_k": args.rrf_k,
                "reranker_variant": args.reranker_variant,
                "local_embedder": embedder_info,
                "bge_model_path": str(args.bge_model_path),
                "e6_dirs": {str(k): str(v) for k, v in seed_dirs.items()},
            },
            "aggregate": aggregate(all_results),
        }
        save_json(args.output_dir / "summary.json", summary)
        progress.log("complete", output_dir=str(args.output_dir))
        print(json.dumps(summary["aggregate"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
