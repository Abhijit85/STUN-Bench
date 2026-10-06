#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_ROOT = Path(os.environ.get("FEDRAG_CANONICAL_ROOT", "<REPO_ROOT>")).resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from math_qa import JinaAIClient
from scripts.run_stabletoolbench_federated import (
    DEFAULT_TOOLBENCH_INSTRUCTION_DIR,
    ProgressLogger,
    apply_junk_filter,
    assert_clean_tree,
    batched_query_embeddings,
    build_tool_registry,
    filter_experience_items,
    load_toolbench_training_items,
    resolve_local_embedder,
    save_json,
    stable_hash,
)

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
DEFAULT_OUTPUT_DIR = CANONICAL_ROOT / "artifacts" / "results" / "toolbench_retrieval_heldout_e7_r1"
DEFAULT_BGE = Path("<HF_CACHE>/models--BAAI--bge-base-en-v1.5/snapshots/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E7 fallback: ToolBench-corpus held-out retrieval-only docs vs shared index.")
    p.add_argument("--toolbench-instruction-dir", type=Path, default=DEFAULT_TOOLBENCH_INSTRUCTION_DIR)
    p.add_argument("--corpora", default="G1,G2,G3")
    p.add_argument("--seeds", default="42,123,456")
    p.add_argument("--holdout-fraction", type=float, default=0.30)
    p.add_argument("--min-tools", type=int, default=300)
    p.add_argument("--min-labels-per-tool", type=int, default=5)
    p.add_argument("--retrievers", default="jina,bm25,bge")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--retrieval-pool-size", type=int, default=20)
    p.add_argument("--embed-model", default="jina-embeddings-v2-base-en")
    p.add_argument("--bge-model-path", type=Path, default=DEFAULT_BGE)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--allow-dirty", action="store_true")
    return p.parse_args()


def parse_csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]


class BM25Index:
    def __init__(self, texts: list[str], *, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.term_counts = [Counter(tokenize(text)) for text in texts]
        self.doc_lens = np.asarray([sum(c.values()) for c in self.term_counts], dtype=np.float32)
        self.avgdl = float(self.doc_lens.mean()) if len(self.doc_lens) else 0.0
        df: Counter[str] = Counter()
        for counts in self.term_counts:
            df.update(counts.keys())
        n_docs = max(1, len(self.term_counts))
        self.idf = {term: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}

    def scores(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.term_counts), dtype=np.float32)
        terms = set(tokenize(query))
        if not terms or not self.term_counts:
            return scores
        avgdl = self.avgdl or 1.0
        for i, counts in enumerate(self.term_counts):
            dl = float(self.doc_lens[i]) or 1.0
            denom_base = self.k1 * (1.0 - self.b + self.b * dl / avgdl)
            total = 0.0
            for term in terms:
                freq = counts.get(term, 0)
                if freq:
                    total += self.idf.get(term, 0.0) * (freq * (self.k1 + 1.0)) / (freq + denom_base)
            scores[i] = total
        return scores


def bge_model(path: Path, device: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(str(path), device=device)


def bge_encode(model: Any, texts: list[str], batch_size: int = 128) -> np.ndarray:
    return np.asarray(model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)


def normalize_matrix(m: np.ndarray) -> np.ndarray:
    if not m.size:
        return m
    norms = np.linalg.norm(m, axis=1)
    norms[norms == 0.0] = 1.0
    return m / norms[:, None]


def tool_doc_text(tool: str, doc: Any) -> str:
    return "\n".join([
        f"tool: {tool}",
        f"categories: {', '.join(doc.categories[:4])}",
        f"apis: {', '.join(doc.api_names[:8])}",
        f"description: {'; '.join(doc.descriptions[:3])}",
    ])


def scenario_text(query: str, tool: str) -> str:
    return f"tool: {tool}\nwhen to use: {query}"


def build_docs(corpus_items, registry, eligible_tools: set[str]) -> list[dict[str, Any]]:
    return [
        {"doc_id": stable_hash({"doc": tool}), "tool": tool, "kind": "description", "text": tool_doc_text(tool, registry[tool])}
        for tool in sorted(eligible_tools)
    ]


def build_shared(corpus_items, registry, eligible_tools: set[str], heldout_tools: set[str]) -> list[dict[str, Any]]:
    docs = build_docs(corpus_items, registry, eligible_tools)
    rows = list(docs)
    for item in corpus_items:
        for tool in item.gold_tools:
            if tool in eligible_tools and tool not in heldout_tools:
                rows.append({"doc_id": stable_hash({"scenario": item.query_id, "tool": tool}), "tool": tool, "kind": "scenario", "text": scenario_text(item.query, tool)})
    return rows


def index_for(retriever: str, docs: list[dict[str, Any]], jina: JinaAIClient, embed_model: str, bge: Any | None) -> tuple[Any, dict[str, Any]]:
    texts = [d["text"] for d in docs]
    if retriever == "bm25":
        idx = BM25Index(texts)
        return idx, {"retriever": "bm25", "tokenizer": "regex:[A-Za-z0-9_]+", "k1": idx.k1, "b": idx.b}
    if retriever == "jina":
        emb = np.asarray(batched_query_embeddings(jina, texts, embed_model), dtype=np.float32)
        return normalize_matrix(emb), {"retriever": "jina", "embed_model": embed_model}
    if retriever == "bge":
        if bge is None:
            raise RuntimeError("BGE requested but model not loaded")
        return bge_encode(bge, texts), {"retriever": "bge", "model_path": "local", "normalized": True}
    raise ValueError(retriever)


def score_query(retriever: str, index: Any, query: str, jina: JinaAIClient, embed_model: str, bge: Any | None) -> np.ndarray:
    if retriever == "bm25":
        return index.scores(query)
    if retriever == "jina":
        q = np.asarray(batched_query_embeddings(jina, [query], embed_model)[0], dtype=np.float32)
        q = q / (np.linalg.norm(q) or 1.0)
        return index @ q if getattr(index, "size", 0) else np.asarray([], dtype=np.float32)
    if retriever == "bge":
        q = bge_encode(bge, [query])[0]
        return index @ q if getattr(index, "size", 0) else np.asarray([], dtype=np.float32)
    raise ValueError(retriever)


def top_distinct_tools(docs: list[dict[str, Any]], scores: np.ndarray, limit: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    if len(scores) == 0:
        return out
    order = np.argsort(-scores)
    for i in order:
        tool = docs[int(i)]["tool"]
        if tool in seen:
            continue
        seen.add(tool)
        out.append(tool)
        if len(out) >= limit:
            break
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "recall_at_5": 0.0, "top1_accuracy": 0.0}
    return {
        "n": len(rows),
        "recall_at_5": sum(bool(r["gold_in_top_5"]) for r in rows) / len(rows),
        "top1_accuracy": sum(bool(r["top1_correct"]) for r in rows) / len(rows),
    }


def mcnemar(rows_docs: list[dict[str, Any]], rows_shared: list[dict[str, Any]]) -> dict[str, Any]:
    by = {r["query_id"]: r for r in rows_shared}
    docs_only = shared_only = compared = 0
    for r in rows_docs:
        s = by.get(r["query_id"])
        if s is None:
            continue
        a = bool(r["gold_in_top_5"])
        b = bool(s["gold_in_top_5"])
        compared += 1
        docs_only += int(a and not b)
        shared_only += int((not a) and b)
    n = docs_only + shared_only
    if n == 0:
        p = 1.0
        method = "exact_binomial"
    elif n <= 1024:
        lo = min(docs_only, shared_only)
        p = min(1.0, 2.0 * sum(math.comb(n, k) for k in range(lo + 1)) / (2 ** n))
        method = "exact_binomial"
    else:
        # Continuity-corrected McNemar chi-square with 1 d.f.; avoids huge integer overflow.
        stat = (abs(docs_only - shared_only) - 1.0) ** 2 / n
        p = math.erfc(math.sqrt(stat / 2.0))
        method = "mcnemar_chi2_cc"
    return {"compared": compared, "docs_only": docs_only, "shared_only": shared_only, "p_value": p, "method": method}


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress = ProgressLogger(args.output_dir)
    t0 = time.perf_counter()
    commit = assert_clean_tree(allow_dirty=args.allow_dirty)
    progress.log("runner_start", repo_commit=commit, fallback_benchmark="ToolBench instruction corpora", elapsed_seconds=0.0)
    embedder = resolve_local_embedder()
    progress.log("embedder_ready", **embedder, elapsed_seconds=round(time.perf_counter()-t0,3))
    jina = JinaAIClient([])
    retrievers = parse_csv(args.retrievers)
    bge = None
    if "bge" in retrievers:
        device = os.environ.get("BGE_DEVICE") or ("cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
        progress.log("load_bge_begin", model_path=str(args.bge_model_path), device=device, elapsed_seconds=round(time.perf_counter()-t0,3))
        bge = bge_model(args.bge_model_path, device)
        progress.log("load_bge_done", model_sha256=sha256_file(args.bge_model_path / "config.json"), elapsed_seconds=round(time.perf_counter()-t0,3))

    all_items = load_toolbench_training_items(args.toolbench_instruction_dir)
    registry_all = build_tool_registry(all_items)
    registry, junk = apply_junk_filter(registry_all)
    items, filt = filter_experience_items(all_items, registry)
    save_json(args.output_dir / "data_filter.json", {"junk_filter": junk, "experience_filter": filt, "item_count": len(items), "registry_count": len(registry)})
    progress.log("data_loaded", item_count=len(items), registry_count=len(registry), elapsed_seconds=round(time.perf_counter()-t0,3))

    corpora = parse_csv(args.corpora)
    seeds = [int(s) for s in parse_csv(args.seeds)]
    all_summary: dict[str, Any] = {}
    for corpus in corpora:
        corpus_items = [it for it in items if it.group == corpus]
        counts = Counter(tool for it in corpus_items for tool in it.gold_tools if tool in registry)
        eligible_tools = {tool for tool, n in counts.items() if n >= args.min_labels_per_tool}
        eval_items = [it for it in corpus_items if any(tool in eligible_tools for tool in it.gold_tools)]
        if len(eligible_tools) < args.min_tools:
            progress.log("corpus_skipped", corpus=corpus, eligible_tool_count=len(eligible_tools), eval_query_count=len(eval_items), elapsed_seconds=round(time.perf_counter()-t0,3))
            all_summary[corpus] = {"skipped": True, "eligible_tool_count": len(eligible_tools), "eval_query_count": len(eval_items)}
            continue
        corpus_summary = {"eligible_tool_count": len(eligible_tools), "eval_query_count": len(eval_items), "seeds": {}}
        for seed in seeds:
            rng = random.Random(stable_hash({"corpus": corpus, "seed": seed})[:16])
            tools = sorted(eligible_tools)
            rng.shuffle(tools)
            heldout_count = max(1, round(len(tools) * args.holdout_fraction))
            heldout = set(tools[:heldout_count])
            heldout_sha = stable_hash(sorted(heldout))
            docs = build_docs(corpus_items, registry, eligible_tools)
            shared = build_shared(corpus_items, registry, eligible_tools, heldout)
            docs_sha = stable_hash([{k:d[k] for k in ("doc_id","tool","kind","text")} for d in docs])
            shared_sha = stable_hash([{k:d[k] for k in ("doc_id","tool","kind","text")} for d in shared])
            progress.log("seed_begin", corpus=corpus, seed=seed, heldout_tool_count=len(heldout), eval_query_count=len(eval_items), docs_doc_count=len(docs), shared_doc_count=len(shared), elapsed_seconds=round(time.perf_counter()-t0,3))
            seed_summary = {"heldout_tools_sha256": heldout_sha, "docs_index_sha256": docs_sha, "shared_index_sha256": shared_sha, "heldout_tool_count": len(heldout), "docs_doc_count": len(docs), "shared_doc_count": len(shared), "retrievers": {}}
            for retriever in retrievers:
                progress.log("retriever_begin", corpus=corpus, seed=seed, retriever=retriever, elapsed_seconds=round(time.perf_counter()-t0,3))
                docs_index, docs_meta = index_for(retriever, docs, jina, args.embed_model, bge)
                shared_index, shared_meta = index_for(retriever, shared, jina, args.embed_model, bge)
                rows_by_arm: dict[str, list[dict[str, Any]]] = {}
                for arm, arm_docs, arm_index in [("docs_only", docs, docs_index), ("shared", shared, shared_index)]:
                    rows = []
                    for qi, item in enumerate(eval_items):
                        gold = [tool for tool in item.gold_tools if tool in eligible_tools]
                        subset = "heldout" if any(tool in heldout for tool in gold) else "labeled"
                        scores = score_query(retriever, arm_index, item.query, jina, args.embed_model, bge)
                        candidates = top_distinct_tools(arm_docs, scores, args.retrieval_pool_size)
                        top5 = candidates[:args.top_k]
                        rows.append({
                            "query_id": item.query_id,
                            "query_text": item.query,
                            "corpus": corpus,
                            "gold_tools": gold,
                            "subset": subset,
                            "candidate_ids": top5,
                            "retrieval_pool_tools": candidates,
                            "gold_in_top_5": any(tool in top5 for tool in gold),
                            "top1_correct": bool(candidates and candidates[0] in gold),
                        })
                        if (qi + 1) % 10000 == 0:
                            progress.log("query_progress", corpus=corpus, seed=seed, retriever=retriever, arm=arm, processed=qi+1, elapsed_seconds=round(time.perf_counter()-t0,3))
                    rows_by_arm[arm] = rows
                    by_subset = {subset: summarize([r for r in rows if r["subset"] == subset]) for subset in ("heldout", "labeled")}
                    out = {
                        "paper_eligible": True,
                        "repo_commit": commit,
                        "benchmark": "ToolBench instruction corpora fallback for ToolRet",
                        "corpus": corpus,
                        "seed": seed,
                        "retriever": retriever,
                        "arm": arm,
                        "top_k": args.top_k,
                        "retrieval_pool_size": args.retrieval_pool_size,
                        "holdout_fraction": args.holdout_fraction,
                        "heldout_tools_sha256": heldout_sha,
                        "docs_index_sha256": docs_sha,
                        "shared_index_sha256": shared_sha,
                        "docs_doc_count": len(docs),
                        "shared_doc_count": len(shared),
                        "retriever_metadata": docs_meta if arm == "docs_only" else shared_meta,
                        "summary": summarize(rows),
                        "subset_metrics": by_subset,
                        "rows": rows,
                    }
                    arm_path = args.output_dir / corpus / f"seed_{seed}" / retriever / f"{arm}.json"
                    save_json(arm_path, out)
                    progress.log("arm_done", corpus=corpus, seed=seed, retriever=retriever, arm=arm, heldout_recall_at_5=by_subset["heldout"]["recall_at_5"], labeled_recall_at_5=by_subset["labeled"]["recall_at_5"], elapsed_seconds=round(time.perf_counter()-t0,3))
                mc = mcnemar(rows_by_arm["docs_only"], rows_by_arm["shared"])
                rsum = {
                    "docs_only": summarize(rows_by_arm["docs_only"]),
                    "shared": summarize(rows_by_arm["shared"]),
                    "docs_only_subset": {s: summarize([r for r in rows_by_arm["docs_only"] if r["subset"] == s]) for s in ("heldout", "labeled")},
                    "shared_subset": {s: summarize([r for r in rows_by_arm["shared"] if r["subset"] == s]) for s in ("heldout", "labeled")},
                    "mcnemar_recall_at_5": mc,
                }
                seed_summary["retrievers"][retriever] = rsum
                save_json(args.output_dir / corpus / f"seed_{seed}" / retriever / "summary.json", rsum)
                progress.log("retriever_done", corpus=corpus, seed=seed, retriever=retriever, mcnemar_p=mc["p_value"], elapsed_seconds=round(time.perf_counter()-t0,3))
            corpus_summary["seeds"][str(seed)] = seed_summary
            save_json(args.output_dir / corpus / f"seed_{seed}" / "seed_summary.json", seed_summary)
            progress.log("seed_done", corpus=corpus, seed=seed, elapsed_seconds=round(time.perf_counter()-t0,3))
        all_summary[corpus] = corpus_summary
    aggregate: dict[str, Any] = {}
    for corpus, csum in all_summary.items():
        if csum.get("skipped"):
            continue
        aggregate[corpus] = {}
        for retriever in retrievers:
            vals = defaultdict(list)
            for seed_s, ssum in csum["seeds"].items():
                r = ssum["retrievers"].get(retriever)
                if not r:
                    continue
                vals["docs_heldout_r5"].append(r["docs_only_subset"]["heldout"]["recall_at_5"])
                vals["shared_heldout_r5"].append(r["shared_subset"]["heldout"]["recall_at_5"])
                vals["docs_labeled_r5"].append(r["docs_only_subset"]["labeled"]["recall_at_5"])
                vals["shared_labeled_r5"].append(r["shared_subset"]["labeled"]["recall_at_5"])
            aggregate[corpus][retriever] = {k: {"mean": statistics.mean(v), "sd": statistics.stdev(v) if len(v)>1 else 0.0, "n_seeds": len(v)} for k,v in vals.items() if v}
    final = {"paper_eligible": True, "repo_commit": commit, "benchmark": "ToolBench instruction corpora fallback for ToolRet", "config": vars(args) | {"bge_model_path": str(args.bge_model_path), "output_dir": str(args.output_dir), "toolbench_instruction_dir": str(args.toolbench_instruction_dir)}, "aggregate": aggregate, "corpora": all_summary}
    save_json(args.output_dir / "summary.json", final)
    progress.log("complete", elapsed_seconds=round(time.perf_counter()-t0,3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
