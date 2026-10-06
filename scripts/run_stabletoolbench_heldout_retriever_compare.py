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
    DEFAULT_TOOLBENCH_INSTRUCTION_DIR,
    GROUPS,
    ProgressLogger,
    apply_junk_filter,
    assert_clean_tree,
    assert_no_eval_overlap,
    batched_query_embeddings,
    build_client_package,
    build_ranked_candidates,
    build_tool_registry,
    combine_packages,
    filter_eval_queries,
    filter_experience_items,
    hash_package,
    limit_client_items,
    load_local_backend,
    load_package_file,
    load_stabletoolbench_queries,
    load_toolbench_training_items,
    maybe_cuda_synchronize,
    package_to_candidates,
    remove_exact_eval_overlaps,
    remove_near_duplicate_eval_overlaps,
    resolve_local_embedder,
    save_package,
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
from synapse.knowledge.compendium import KnowledgePackage

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
DEFAULT_OUTPUT_DIR = CANONICAL_ROOT / "artifacts" / "results" / "stabletoolbench_heldout_retriever_compare_e6_r1"
DEFAULT_E5_DIR = CANONICAL_ROOT / "artifacts" / "results" / "stabletoolbench_heldout_r3b" / "seed_42"
DEFAULT_SEED42_R4_PACKAGE_DIR = CANONICAL_ROOT / "artifacts" / "verification" / "stabletoolbench_clean_anchor_seed42_r4" / "packages" / "seed_42"
DEFAULT_BGE = Path("<HF_CACHE>/models--BAAI--bge-base-en-v1.5/snapshots/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E6: held-out shared-vs-description retrieval under alternate retrievers.")
    parser.add_argument("--stb-root", type=Path, default=DEFAULT_STB_ROOT)
    parser.add_argument("--toolbench-instruction-dir", type=Path, default=DEFAULT_TOOLBENCH_INSTRUCTION_DIR)
    parser.add_argument("--groups", type=str, default=",".join(GROUPS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--client-count", type=int, default=5)
    parser.add_argument("--max-items-per-client", type=int, default=5000)
    parser.add_argument("--partition-mode", choices=["category", "iid"], default="category")
    parser.add_argument("--retrievers", type=str, default="jina,bm25,bge")
    parser.add_argument("--arms", type=str, default="docs_only,synapse_shared,synapse_separate")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retrieval-pool-size", type=int, default=20)
    parser.add_argument("--retrieval-mode", type=str, default="distinct_tool_topk")
    parser.add_argument("--reranker-variant", type=str, default="V3")
    parser.add_argument("--embed-model", type=str, default="jina-embeddings-v2-base-en")
    parser.add_argument("--bge-model-path", type=Path, default=DEFAULT_BGE)
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--pool-embedding-cache-dir", type=Path, default=CANONICAL_ROOT / "artifacts" / "cache" / "stabletoolbench")
    parser.add_argument("--contamination-near-duplicate-threshold", type=float, default=0.95)
    parser.add_argument("--contamination-report-threshold", type=float, default=0.90)
    parser.add_argument("--e5-seed-dir", type=Path, default=DEFAULT_E5_DIR)
    parser.add_argument("--flat-package-path", type=Path, default=DEFAULT_SEED42_R4_PACKAGE_DIR / "flat_pool.json")
    parser.add_argument("--tool-doc-package-path", type=Path, default=DEFAULT_SEED42_R4_PACKAGE_DIR / "tool_docs.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args()


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


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

    def encode_query(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.term_counts), dtype=np.float32)
        terms = set(tokenize(query))
        if not terms or not self.term_counts:
            return scores
        avgdl = self.avgdl or 1.0
        for idx, counts in enumerate(self.term_counts):
            dl = float(self.doc_lens[idx]) or 1.0
            denom_base = self.k1 * (1.0 - self.b + self.b * dl / avgdl)
            total = 0.0
            for term in terms:
                freq = counts.get(term, 0)
                if freq:
                    total += self.idf.get(term, 0.0) * (freq * (self.k1 + 1.0)) / (freq + denom_base)
            scores[idx] = total
        return scores


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def mcnemar_recall(rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]]) -> dict[str, Any]:
    by_b = {row["query_id"]: row for row in rows_b}
    b01 = b10 = compared = 0
    for row_a in rows_a:
        row_b = by_b.get(row_a["query_id"])
        if row_b is None:
            continue
        a = bool(row_a.get("gold_in_top_k"))
        b = bool(row_b.get("gold_in_top_k"))
        compared += 1
        b10 += int(a and not b)
        b01 += int((not a) and b)
    n = b01 + b10
    if n == 0:
        p_value = 1.0
    else:
        # Exact two-sided binomial test under p=0.5.
        lo = min(b01, b10)
        p_value = min(1.0, 2.0 * sum(math.comb(n, k) for k in range(lo + 1)) / (2**n))
    return {"compared": compared, "b10_a_only": b10, "b01_b_only": b01, "p_value": p_value}


def build_bge_encoder(model_path: Path, device: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(str(model_path), device=device)


def bge_encode(model: Any, texts: list[str], *, batch_size: int = 128) -> np.ndarray:
    return np.asarray(
        model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False),
        dtype=np.float32,
    )


def normalized_matrix(vectors: list[list[float]]) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.size:
        norms = np.linalg.norm(matrix, axis=1)
        norms[norms == 0.0] = 1.0
        matrix = matrix / norms[:, None]
    return matrix


def prepare_candidates(package: KnowledgePackage, retriever: str, jina_client: JinaAIClient, embed_model: str, bge_model: Any | None):
    candidates = package_to_candidates(package, jina_client, embed_model)
    texts = [candidate.text for candidate in candidates]
    if retriever == "jina":
        index = normalized_matrix([candidate.embedding for candidate in candidates]) if candidates else np.zeros((0, 0), dtype=np.float32)
        meta = {"retriever": "jina", "embed_model": embed_model}
    elif retriever == "bge":
        if bge_model is None:
            raise RuntimeError("BGE model requested but not loaded")
        index = bge_encode(bge_model, texts)
        meta = {"retriever": "bge", "model_path": str(getattr(bge_model, "_model_card_text", "") or "")}
    elif retriever == "bm25":
        index = BM25Index(texts)
        meta = {"retriever": "bm25", "tokenizer": "regex:[A-Za-z0-9_]+", "k1": index.k1, "b": index.b}
    else:
        raise ValueError(f"unknown retriever: {retriever}")
    return candidates, index, meta


def retrieve(candidates, index, retriever: str, query: str, query_embedding: list[float] | None, bge_model: Any | None, args):
    if retriever == "bm25":
        scores = index.encode_query(query)
    elif retriever == "bge":
        q = bge_encode(bge_model, [query])[0]
        scores = index @ q if getattr(index, "size", 0) else np.asarray([], dtype=np.float32)
    else:
        q = np.asarray(query_embedding, dtype=np.float32)
        qn = float(np.linalg.norm(q))
        if qn > 0.0:
            q = q / qn
        scores = index @ q if getattr(index, "size", 0) else np.asarray([], dtype=np.float32)
    return build_ranked_candidates(candidates, scores, args.retrieval_pool_size, args.top_k, args.retrieval_mode)


def summarize_subset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "accuracy": sum(1 for row in rows if row["routed_correctly"]) / len(rows) if rows else 0.0,
        "recall_at_5": sum(1 for row in rows if row["gold_in_top_k"]) / len(rows) if rows else 0.0,
        "retrieval_top1": sum(1 for row in rows if (row.get("top_candidates") or [""])[0] in set(row.get("gold_parent_tools") or [])) / len(rows) if rows else 0.0,
    }


def add_subsets(result: dict[str, Any]) -> None:
    for subset in ("heldout", "labeled"):
        bucket = [row for row in result["rows"] if row.get("subset") == subset]
        result.setdefault("subset_metrics", {})[subset] = summarize_subset(bucket)


def evaluate_arm(
    *,
    arm: str,
    retriever: str,
    retrieval_package: KnowledgePackage,
    rerank_package: KnowledgePackage,
    test_items,
    heldout_tools: set[str],
    jina_client: JinaAIClient,
    embed_model: str,
    bge_model: Any | None,
    backend: Any,
    args,
    progress: ProgressLogger,
):
    progress.log("arm_prepare_begin", retriever=retriever, arm=arm, retrieval_artifact_count=len(retrieval_package.artifacts), rerank_artifact_count=len(rerank_package.artifacts))
    candidates, index, retriever_meta = prepare_candidates(retrieval_package, retriever, jina_client, embed_model, bge_model)
    rerank_by_tool = {}
    if rerank_package is retrieval_package:
        rerank_by_tool = None
    else:
        for candidate in package_to_candidates(rerank_package, jina_client, embed_model):
            rerank_by_tool.setdefault(candidate.parent_tool, candidate)
    query_embeddings = batched_query_embeddings(jina_client, [item.query for item in test_items], embed_model) if retriever == "jina" else [None] * len(test_items)
    rows = []
    for idx, (item, query_embedding) in enumerate(zip(test_items, query_embeddings), start=1):
        started = time.perf_counter()
        ranked, pool_tools = retrieve(candidates, index, retriever, item.query, query_embedding, bge_model, args)
        retrieval_ranked = ranked
        if rerank_by_tool is not None:
            mapped = []
            missing = []
            for candidate in ranked:
                replacement = rerank_by_tool.get(candidate.parent_tool)
                if replacement is None:
                    missing.append(candidate.parent_tool)
                else:
                    mapped.append(replacement)
            ranked = mapped
        else:
            missing = []
        top_candidate = ranked[0] if ranked else None
        if top_candidate is None:
            predicted = ""
            parse_ok = False
            fallback = True
            prompt_hash = ""
            rerank_s = 0.0
        else:
            maybe_cuda_synchronize(backend)
            rerank_started = time.perf_counter()
            result = run_prompt(backend, args.reranker_variant, "toolbench", item.query, ranked, [], top_candidate)
            maybe_cuda_synchronize(backend)
            rerank_s = time.perf_counter() - rerank_started
            predicted = result.predicted_tool
            parse_ok = result.parse_ok
            fallback = result.fallback_used
            prompt_hash = result.prompt_hash
        gold = item.gold_tools
        subset = "heldout" if gold and all(tool in heldout_tools for tool in gold) else "labeled"
        rows.append(
            {
                "query_id": item.query_id,
                "query_text": item.query,
                "group": item.group,
                "gold_tools": gold,
                "gold_parent_tools": gold,
                "candidate_ids": [candidate.candidate_id for candidate in retrieval_ranked],
                "candidate_tools": [candidate.parent_tool for candidate in retrieval_ranked],
                "top_candidate_ids": [candidate.candidate_id for candidate in ranked],
                "top_candidates": [candidate.parent_tool for candidate in ranked],
                "retrieval_pool_tools": pool_tools,
                "predicted_tool": predicted,
                "routed_correctly": predicted in gold,
                "correct": predicted in gold,
                "gold_in_top_k": any(tool in gold for tool in [candidate.parent_tool for candidate in retrieval_ranked]),
                "subset": subset,
                "parse_ok": parse_ok,
                "fallback_used": fallback,
                "missing_rerank_tools": missing,
                "total_s": time.perf_counter() - started,
                "rerank_s": rerank_s,
                "prompt_hash": prompt_hash,
            }
        )
        if idx == 1 or idx % 25 == 0 or idx == len(test_items):
            progress.log("arm_progress", retriever=retriever, arm=arm, completed_queries=idx, total_queries=len(test_items), running_accuracy=sum(1 for row in rows if row["routed_correctly"]) / len(rows), running_recall_at_5=sum(1 for row in rows if row["gold_in_top_k"]) / len(rows))
    result = summarize_rows(rows)
    result.update({"arm": arm, "retriever": retriever, "rows": rows, "retriever_metadata": retriever_meta})
    add_subsets(result)
    return result


def aggregate(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for retriever in sorted({key.split("/", 1)[0] for key in results}):
        out[retriever] = {}
        for arm in sorted(key.split("/", 1)[1] for key in results if key.startswith(retriever + "/")):
            result = results[f"{retriever}/{arm}"]
            out[retriever][arm] = {
                "accuracy": result["accuracy"],
                "recall_at_5": result["recall_at_5"],
                "heldout": result["subset_metrics"]["heldout"],
                "labeled": result["subset_metrics"]["labeled"],
            }
        if f"{retriever}/docs_only" in results and f"{retriever}/synapse_shared" in results:
            docs = results[f"{retriever}/docs_only"]
            shared = results[f"{retriever}/synapse_shared"]
            out[retriever]["shared_minus_docs_heldout_recall"] = shared["subset_metrics"]["heldout"]["recall_at_5"] - docs["subset_metrics"]["heldout"]["recall_at_5"]
            out[retriever]["shared_vs_docs_mcnemar_heldout_recall"] = mcnemar_recall(
                [row for row in shared["rows"] if row["subset"] == "heldout"],
                [row for row in docs["rows"] if row["subset"] == "heldout"],
            )
    return out


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    commit, dirty = assert_clean_tree(allow_dirty=args.allow_dirty)
    progress = ProgressLogger(args.output_dir)
    progress.log("launch", repo_commit=commit, dirty_entry_count=len(dirty))
    retrievers = parse_csv(args.retrievers)
    arms = parse_csv(args.arms)
    groups = parse_csv(args.groups)
    embedder_info = resolve_local_embedder()
    jina_client = JinaAIClient(api_keys=[])

    with temporary_env({"JINA_LOCAL_EMBED_MODEL": embedder_info["model_path"], "JINA_LOCAL_EMBED_DEVICE": embedder_info["device"], "JINA_LOCAL_EMBED_LOCAL_ONLY": embedder_info["local_only"], "JINA_API_KEY": None}):
        all_queries = load_stabletoolbench_queries(args.stb_root, groups)
        registry = build_tool_registry(all_queries)
        train_items = load_toolbench_training_items(args.toolbench_instruction_dir)
        registry.update(build_tool_registry(train_items))
        registry, junk_info = apply_junk_filter(registry)
        queries, eval_filter = filter_eval_queries(all_queries, registry)
        train_items, train_filter = filter_experience_items(train_items, registry)
        heldout_tools, heldout_sanity = heldout_tool_set(queries, HELDOUT_GROUPS)
        heldout_set = set(heldout_tools)
        heldout_sha = stable_hash(heldout_tools)
        progress.log("data_ready", test_count=len(queries), train_count=len(train_items), heldout_tool_count=len(heldout_tools), heldout_tools_sha256=heldout_sha, junk_filter=junk_info, eval_filter=eval_filter, train_filter=train_filter)
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
        leak_filter = {**exact_info, **near_info}
        leak_filter["pool_items_removed_total"] = int(leak_filter["pool_items_removed_exact"]) + int(leak_filter["pool_items_removed_near_dup"])
        leak_filter["eval_queries_removed"] = 0
        leak_filter["post_filter_refusal_check"] = assert_no_eval_overlap(train_items, queries, forbidden_near_duplicate_texts=forbidden_near_duplicate_texts, stage="e6_post_leak_filter_pool")
        progress.log("contamination_filter_done", train_count=len(train_items), test_count=len(queries), **leak_filter)
        train_items, heldout_filter = filter_heldout_pool(train_items, heldout_set)
        if heldout_filter["remaining_items_with_heldout_label"] != 0:
            raise RuntimeError("heldout label filter failed")
        heldout_filter["eval_queries_removed"] = 0
        heldout_filter["heldout_tools_sha256"] = heldout_sha
        heldout_filter["pool_sha256"] = stable_hash([{"query_id": item.query_id, "query": item.query, "gold_tools": item.gold_tools} for item in train_items])
        progress.log("heldout_pool_filter_done", **heldout_filter)

        clients = limit_client_items(assign_clients(train_items, args.client_count, args.partition_mode, args.seed), args.max_items_per_client, args.seed)
        tool_doc_package, tool_doc_hash = load_package_file(args.tool_doc_package_path)
        flat_package, flat_hash = load_package_file(args.flat_package_path)
        e5_flat = json.loads((args.e5_seed_dir / "flat_pool.json").read_text())
        e5_synapse = json.loads((args.e5_seed_dir / "synapse.json").read_text())
        e5_flat_hash = e5_flat["compendium"]["global_sha256"]
        if flat_hash != e5_flat_hash:
            raise RuntimeError(f"flat package hash mismatch: loaded {flat_hash} != E5 {e5_flat_hash}")
        e5_tool_doc_hash = e5_synapse.get("compendium", {}).get("tool_doc_sha256")
        if e5_tool_doc_hash and tool_doc_hash != e5_tool_doc_hash:
            raise RuntimeError(f"tool-doc package hash mismatch: loaded {tool_doc_hash} != E5 {e5_tool_doc_hash}")
        progress.log(
            "persisted_docs_loaded",
            flat_package_path=str(args.flat_package_path),
            flat_pool_sha256=flat_hash,
            tool_doc_package_path=str(args.tool_doc_package_path),
            tool_doc_sha256=tool_doc_hash,
            e5_recorded_synapse_sha256=e5_synapse["compendium"]["global_sha256"],
        )
        client_packages = []
        client_hashes = []
        package_dir = args.output_dir / f"seed_{args.seed}" / "packages"
        package_dir.mkdir(parents=True, exist_ok=True)
        for client_id, items in sorted(clients.items()):
            progress.log("client_package_begin", seed=args.seed, client_id=client_id, item_count=len(items))
            package, package_hash = build_client_package(client_id, items, registry, jina_client, args.embed_model)
            save_package(package_dir / f"{client_id}.json", package, package_hash)
            client_packages.append(package)
            client_hashes.append(package_hash)
            progress.log("client_package_done", seed=args.seed, client_id=client_id, artifact_count=len(package.artifacts), package_sha256=package_hash)
        progress.log("synapse_merge_begin", seed=args.seed)
        with temporary_env({"SYNAPSE_EDGE_MERGE_POLICY": "conflict_log"}):
            merged = EdgeAggregator(EdgeConfig(edge_id=f"stabletoolbench_e6_seed_{args.seed}")).merge_packages(client_packages)
        if merged is None:
            raise RuntimeError("synapse merge produced no package")
        synapse_package, synapse_hash = combine_packages("synapse_with_docs", [tool_doc_package, merged])
        save_package(package_dir / "flat_pool.json", flat_package, flat_hash)
        save_package(package_dir / "tool_docs.json", tool_doc_package, tool_doc_hash)
        save_package(package_dir / "synapse_shared.json", synapse_package, synapse_hash)
        progress.log("synapse_merge_done", seed=args.seed, synapse_sha256=synapse_hash, e5_recorded_synapse_sha256=json.loads((args.e5_seed_dir / "synapse.json").read_text())["compendium"]["global_sha256"], artifact_count=len(synapse_package.artifacts))

        progress.log("load_backend_begin", model_path=args.model_path)
        backend = load_local_backend(args.model_path)
        progress.log("load_backend_done")
        bge_model = None
        if "bge" in retrievers:
            progress.log("load_bge_begin", model_path=str(args.bge_model_path))
            device = os.environ.get("BGE_DEVICE") or ("cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
            bge_model = build_bge_encoder(args.bge_model_path, device=device)
            progress.log("load_bge_done", model_path=str(args.bge_model_path), model_sha256=sha256_file(args.bge_model_path / "config.json"), device=device)

        results: dict[str, dict[str, Any]] = {}
        for retriever in retrievers:
            for arm in arms:
                retrieval_package = flat_package if arm in {"docs_only", "synapse_separate"} else synapse_package
                rerank_package = synapse_package if arm == "synapse_separate" else retrieval_package
                result = evaluate_arm(
                    arm=arm,
                    retriever=retriever,
                    retrieval_package=retrieval_package,
                    rerank_package=rerank_package,
                    test_items=queries,
                    heldout_tools=heldout_set,
                    jina_client=jina_client,
                    embed_model=args.embed_model,
                    bge_model=bge_model,
                    backend=backend,
                    args=args,
                    progress=progress,
                )
                result.update({
                    "paper_eligible": True,
                    "repo_commit": commit,
                    "data_mode": "toolbench_train",
                    "seed": args.seed,
                    "heldout_tools_sha256": heldout_sha,
                    "heldout_sanity": heldout_sanity,
                    "heldout_filter": heldout_filter,
                    "contamination_filter": leak_filter,
                    "source": {
                        "e5_seed_dir": str(args.e5_seed_dir),
                        "reconstructed_synapse_from_e5_filter_config": True,
                        "e5_recorded_synapse_sha256": e5_synapse["compendium"]["global_sha256"],
                        "flat_package_path": str(args.flat_package_path),
                        "tool_doc_package_path": str(args.tool_doc_package_path),
                        "flat_pool_sha256": flat_hash,
                        "synapse_sha256": synapse_hash,
                        "tool_doc_sha256": tool_doc_hash,
                        "client_sha256": client_hashes,
                    },
                    "index_document_counts": {"docs_only": len(flat_package.artifacts), "synapse_shared": len(synapse_package.artifacts)},
                    "metric_definition": {"correct": "predicted_tool in gold_parent_tools", "recall_at_5": "any gold tool appears among retrieved candidate tools"},
                })
                if arm == "synapse_separate":
                    docs_rows = results.get(f"{retriever}/docs_only", {}).get("rows", [])
                    if docs_rows:
                        mismatches = [
                            row["query_id"]
                            for row, docs_row in zip(result["rows"], docs_rows)
                            if row.get("candidate_ids") != docs_row.get("candidate_ids")
                        ]
                        result["candidate_ids_identical_to_docs_only"] = len(mismatches) == 0
                        result["candidate_id_mismatch_count"] = len(mismatches)
                        if mismatches:
                            raise RuntimeError(f"synapse_separate candidate ids diverged from docs_only for {len(mismatches)} queries")
                out_path = args.output_dir / f"seed_{args.seed}" / retriever / f"{arm}.json"
                save_json(out_path, result)
                results[f"{retriever}/{arm}"] = result
                progress.log("arm_done", retriever=retriever, arm=arm, accuracy=result["accuracy"], recall_at_5=result["recall_at_5"], heldout_recall_at_5=result["subset_metrics"]["heldout"]["recall_at_5"])
        summary = {
            "paper_eligible": True,
            "config": {
                "repo_commit": commit,
                "seed": args.seed,
                "retrievers": retrievers,
                "arms": arms,
                "heldout_tools_sha256": heldout_sha,
                "retrieval_pool_size": args.retrieval_pool_size,
                "top_k": args.top_k,
                "reranker_variant": args.reranker_variant,
                "local_embedder": embedder_info,
                "bge_model_path": str(args.bge_model_path),
                "e5_seed_dir": str(args.e5_seed_dir),
            },
            "aggregate": aggregate(results),
        }
        save_json(args.output_dir / "summary.json", summary)
        progress.log("complete", output_dir=str(args.output_dir))
        print(json.dumps(summary["aggregate"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
