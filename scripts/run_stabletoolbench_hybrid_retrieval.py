#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
from collections import Counter
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
    assert_clean_tree,
    batched_query_embeddings,
    hash_package,
    load_local_backend,
    load_package_file,
    maybe_cuda_synchronize,
    package_to_candidates,
    resolve_local_embedder,
    save_json,
    summarize_rows,
    temporary_env,
)
from scripts.run_stabletoolbench_typing_isolation import load_eval_queries, load_seed_packages

DEFAULT_OUTPUT_DIR = CANONICAL_ROOT / "artifacts" / "results" / "stabletoolbench_hybrid_retrieval_seed456_r1"
DEFAULT_PACKAGES_ROOT = CANONICAL_ROOT / "artifacts" / "verification" / "stabletoolbench_clean_anchor_seed456_r4" / "packages"

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="StableToolBench BM25+Jina RRF retrieval control.")
    parser.add_argument("--seed", type=int, default=456)
    parser.add_argument("--arms", type=str, default="synapse,flat_pool")
    parser.add_argument("--groups", type=str, default=",".join(GROUPS))
    parser.add_argument("--packages-root", type=Path, default=DEFAULT_PACKAGES_ROOT)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retrieval-pool-size", type=int, default=20)
    parser.add_argument("--rrf-k", type=float, default=60.0)
    parser.add_argument("--bm25-k1", type=float, default=1.2)
    parser.add_argument("--bm25-b", type=float, default=0.75)
    parser.add_argument("--reranker-variant", type=str, default="V3")
    parser.add_argument("--embed-model", type=str, default="jina-embeddings-v2-base-en")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--stb-root", type=Path, default=DEFAULT_STB_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args()


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


class BM25Index:
    def __init__(self, texts: list[str], *, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.term_counts = [Counter(tokenize(text)) for text in texts]
        self.doc_lens = np.asarray([sum(counts.values()) for counts in self.term_counts], dtype=np.float32)
        self.avgdl = float(self.doc_lens.mean()) if len(self.doc_lens) else 0.0
        df: Counter[str] = Counter()
        for counts in self.term_counts:
            df.update(counts.keys())
        n_docs = max(1, len(self.term_counts))
        self.idf = {term: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}

    def scores(self, query: str) -> np.ndarray:
        query_terms = set(tokenize(query))
        scores = np.zeros(len(self.term_counts), dtype=np.float32)
        if not query_terms or not self.term_counts:
            return scores
        avgdl = self.avgdl or 1.0
        for idx, counts in enumerate(self.term_counts):
            dl = float(self.doc_lens[idx]) or 1.0
            denom_base = self.k1 * (1.0 - self.b + self.b * dl / avgdl)
            total = 0.0
            for term in query_terms:
                freq = counts.get(term, 0)
                if not freq:
                    continue
                total += self.idf.get(term, 0.0) * (freq * (self.k1 + 1.0)) / (freq + denom_base)
            scores[idx] = total
        return scores


def select_hybrid_candidates(candidates, dense_scores: np.ndarray, bm25_scores: np.ndarray, *, retrieval_pool_size: int, top_k: int, rrf_k: float):
    if not candidates:
        return [], []
    dense_order = np.argsort(-dense_scores).tolist()
    bm25_order = np.argsort(-bm25_scores).tolist()
    rrf = np.zeros(len(candidates), dtype=np.float32)
    for rank, idx in enumerate(dense_order):
        rrf[idx] += 1.0 / (rrf_k + rank + 1)
    for rank, idx in enumerate(bm25_order):
        rrf[idx] += 1.0 / (rrf_k + rank + 1)
    ranked_indices = np.argsort(-rrf).tolist()
    pool_indices = ranked_indices[:max(top_k, retrieval_pool_size)]
    pool_tools = [candidates[idx].parent_tool for idx in pool_indices]
    selected_tools: list[str] = []
    seen: set[str] = set()
    for idx in pool_indices:
        tool = candidates[idx].parent_tool
        if tool in seen:
            continue
        seen.add(tool)
        selected_tools.append(tool)
        if len(selected_tools) >= top_k:
            break
    selected_set = set(selected_tools)
    expanded = []
    counts = {tool: 0 for tool in selected_tools}
    for idx in ranked_indices:
        candidate = candidates[idx]
        if candidate.parent_tool not in selected_set:
            continue
        if counts[candidate.parent_tool] >= 2:
            continue
        expanded.append(candidate)
        counts[candidate.parent_tool] += 1
    return expanded, pool_tools


def evaluate_hybrid_arm(name: str, package, test_items, jina_client, embed_model: str, backend: Any, args, *, progress: ProgressLogger):
    progress.log("arm_prepare_begin", seed=args.seed, arm=name, artifact_count=len(package.artifacts), query_count=len(test_items))
    prepare_started = time.perf_counter()
    candidates = package_to_candidates(package, jina_client, embed_model)
    bm25 = BM25Index([candidate.text for candidate in candidates], k1=args.bm25_k1, b=args.bm25_b)
    candidate_matrix = np.asarray([candidate.embedding for candidate in candidates], dtype=np.float32) if candidates else np.zeros((0, 0), dtype=np.float32)
    if candidate_matrix.size:
        norms = np.linalg.norm(candidate_matrix, axis=1)
        norms[norms == 0.0] = 1.0
        candidate_matrix = candidate_matrix / norms[:, None]
    progress.log("arm_candidates_ready", seed=args.seed, arm=name, candidate_count=len(candidates), elapsed_prepare_s=time.perf_counter() - prepare_started)
    query_started = time.perf_counter()
    query_embeddings = batched_query_embeddings(jina_client, [item.query for item in test_items], embed_model) if test_items else []
    progress.log("arm_queries_embedded", seed=args.seed, arm=name, query_count=len(query_embeddings), elapsed_query_embed_s=time.perf_counter() - query_started)

    rows: list[dict[str, Any]] = []
    total_latency = total_retrieval = total_rerank = 0.0
    parse_failures = 0
    for idx, (item, embedding) in enumerate(zip(test_items, query_embeddings), start=1):
        total_started = time.perf_counter()
        retrieval_started = time.perf_counter()
        query_vector = np.asarray(embedding, dtype=np.float32)
        norm = float(np.linalg.norm(query_vector))
        if norm > 0.0:
            query_vector = query_vector / norm
        dense_scores = candidate_matrix @ query_vector if candidate_matrix.size else np.asarray([], dtype=np.float32)
        bm25_scores = bm25.scores(item.query)
        ranked, retrieval_pool_tools = select_hybrid_candidates(
            candidates,
            dense_scores,
            bm25_scores,
            retrieval_pool_size=args.retrieval_pool_size,
            top_k=args.top_k,
            rrf_k=args.rrf_k,
        )
        retrieval_s = time.perf_counter() - retrieval_started
        top_candidate = ranked[0] if ranked else None
        if top_candidate is None:
            total_s = time.perf_counter() - total_started
            rows.append({"query_id": item.query_id, "query_text": item.query, "group": item.group, "gold_parent_tools": item.gold_tools, "predicted_tool": "", "routed_correctly": False, "gold_in_top_k": False, "top_candidates": [], "top_candidate_ids": [], "retrieval_pool_tools": retrieval_pool_tools, "parse_ok": False, "fallback_used": True, "latency_seconds": total_s, "retrieval_s": retrieval_s, "rerank_s": 0.0, "total_s": total_s})
            total_latency += total_s
            total_retrieval += retrieval_s
            continue
        maybe_cuda_synchronize(backend)
        rerank_started = time.perf_counter()
        result = run_prompt(backend, args.reranker_variant, "toolbench", item.query, ranked, [], top_candidate)
        maybe_cuda_synchronize(backend)
        rerank_s = time.perf_counter() - rerank_started
        total_s = time.perf_counter() - total_started
        total_latency += total_s
        total_retrieval += retrieval_s
        total_rerank += rerank_s
        parse_failures += int(not result.parse_ok)
        candidate_tools = [candidate.parent_tool for candidate in ranked]
        rows.append({
            "query_id": item.query_id,
            "query_text": item.query,
            "group": item.group,
            "gold_parent_tools": item.gold_tools,
            "predicted_tool": result.predicted_tool,
            "predicted_candidate": result.predicted_candidate,
            "routed_correctly": result.predicted_tool in item.gold_tools,
            "gold_in_top_k": any(tool in item.gold_tools for tool in retrieval_pool_tools),
            "top_candidates": candidate_tools,
            "top_candidate_ids": [candidate.candidate_id for candidate in ranked],
            "retrieval_pool_tools": retrieval_pool_tools,
            "parse_ok": result.parse_ok,
            "fallback_used": result.fallback_used,
            "latency_seconds": total_s,
            "retrieval_s": retrieval_s,
            "rerank_s": rerank_s,
            "total_s": total_s,
            "prompt_hash": result.prompt_hash,
        })
        if idx == 1 or idx % 25 == 0 or idx == len(test_items):
            progress.log("arm_progress", seed=args.seed, arm=name, completed_queries=idx, total_queries=len(test_items), latest_query_id=item.query_id, running_accuracy=sum(1 for row in rows if row["routed_correctly"]) / len(rows), running_recall_at_5=sum(1 for row in rows if row["gold_in_top_k"]) / len(rows), mean_total_s=total_latency / len(rows), mean_retrieval_s=total_retrieval / len(rows), mean_rerank_s=total_rerank / len(rows))
    summary = summarize_rows(rows)
    summary.update({
        "arm": name,
        "rows": rows,
        "parse_failure_rate": parse_failures / len(test_items) if test_items else 0.0,
        "retrieval_backend": "hybrid_rrf_bm25_jina",
        "retrieval_mode": "hybrid_rrf_distinct_tool_topk",
        "retrieval_pool_size": args.retrieval_pool_size,
        "top_k": args.top_k,
        "rrf_k": args.rrf_k,
        "bm25_k1": args.bm25_k1,
        "bm25_b": args.bm25_b,
        "candidate_count": len(candidates),
        "mean_latency_seconds": total_latency / len(test_items) if test_items else 0.0,
        "mean_retrieval_seconds": total_retrieval / len(test_items) if test_items else 0.0,
        "mean_rerank_seconds": total_rerank / len(test_items) if test_items else 0.0,
    })
    return summary


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    commit, dirty = assert_clean_tree(allow_dirty=args.allow_dirty)
    progress = ProgressLogger(args.output_dir)
    progress.log("launch", git_commit=commit, dirty_entry_count=len(dirty))
    embedder = resolve_local_embedder()
    jina_client = JinaAIClient(api_keys=os.environ.get("JINA_API_KEY") and [os.environ["JINA_API_KEY"]] or [])
    arms = parse_csv(args.arms)
    groups = parse_csv(args.groups)
    seed_dir = args.packages_root / f"seed_{args.seed}"
    _client_packages, tool_doc_package, tool_doc_hash = load_seed_packages(args.packages_root, args.seed)
    test_items = load_eval_queries(args.stb_root, groups, tool_doc_package)
    arm_paths = {
        "synapse": seed_dir / "synapse_conflictlog_cached.json",
        "centralized": seed_dir / "centralized_cached.json",
        "flat_pool": seed_dir / "flat_pool.json",
    }
    progress.log("setup_done", seed=args.seed, arms=arms, groups=groups, query_count=len(test_items), packages_root=str(args.packages_root))
    results: dict[str, Any] = {}
    with temporary_env({
        "JINA_LOCAL_EMBED_MODEL": embedder["model_path"],
        "JINA_LOCAL_EMBED_DEVICE": embedder["device"],
        "JINA_LOCAL_EMBED_LOCAL_ONLY": embedder["local_only"],
        "JINA_API_KEY": None,
    }):
        backend = load_local_backend(args.model_path)
        for arm in arms:
            if arm not in arm_paths:
                raise ValueError(f"unsupported hybrid arm: {arm}; supported={sorted(arm_paths)}")
            package, package_hash = load_package_file(arm_paths[arm])
            progress.log("arm_begin", arm=arm, package_path=str(arm_paths[arm]), package_sha256=package_hash)
            result = evaluate_hybrid_arm(arm, package, test_items, jina_client, args.embed_model, backend, args, progress=progress)
            result.update({
                "seed": args.seed,
                "package_path": str(arm_paths[arm]),
                "package_sha256": package_hash,
                "tool_doc_sha256": tool_doc_hash,
                "embedder": embedder,
                "git_commit": commit,
                "paper_eligible": True,
                "metric_definition": {
                    "correct": "predicted_tool in gold_parent_tools",
                    "recall_at_5": "any gold tool present among retrieval pool tools",
                },
            })
            save_json(args.output_dir / f"{arm}.json", result)
            results[arm] = {"accuracy": result["accuracy"], "recall_at_5": result["recall_at_5"], "group_metrics": result["group_metrics"]}
            progress.log("arm_done", arm=arm, accuracy=result["accuracy"], recall_at_5=result["recall_at_5"])
    summary = {
        "paper_eligible": True,
        "config": {
            "git_commit": commit,
            "seed": args.seed,
            "arms": arms,
            "groups": groups,
            "packages_root": str(args.packages_root),
            "stb_root": str(args.stb_root),
            "retrieval_backend": "hybrid_rrf_bm25_jina",
            "retrieval_mode": "hybrid_rrf_distinct_tool_topk",
            "retrieval_pool_size": args.retrieval_pool_size,
            "top_k": args.top_k,
            "rrf_k": args.rrf_k,
            "bm25_k1": args.bm25_k1,
            "bm25_b": args.bm25_b,
            "embedder": embedder,
        },
        "results": results,
    }
    save_json(args.output_dir / "summary.json", summary)
    progress.log("complete", output_dir=str(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
