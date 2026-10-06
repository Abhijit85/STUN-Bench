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
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_ROOT = Path(os.environ.get("FEDRAG_CANONICAL_ROOT", REPO_ROOT)).resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from math_qa import JinaAIClient
from scripts.run_stabletoolbench_federated import (
    ProgressLogger,
    assert_clean_tree,
    batched_query_embeddings,
    resolve_local_embedder,
    save_json,
    stable_hash,
)

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
DEFAULT_BGE = Path("<HF_CACHE>/models--BAAI--bge-base-en-v1.5/snapshots/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E7 ToolRet: per-tool query split, held-out docs vs shared retrieval.")
    p.add_argument("--categories", default="Data,Finance,Sports")
    p.add_argument("--seeds", default="42,123,456")
    p.add_argument("--retrievers", default="bm25,jina,bge")
    p.add_argument("--holdout-fraction", type=float, default=0.30)
    p.add_argument("--min-tools", type=int, default=300)
    p.add_argument("--min-queries-per-tool", type=int, default=5)
    p.add_argument("--scenario-fraction", type=float, default=0.50)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--retrieval-pool-size", type=int, default=20)
    p.add_argument("--near-duplicate-threshold", type=float, default=0.95)
    p.add_argument("--embed-model", default="jina-embeddings-v2-base-en")
    p.add_argument("--bge-model-path", type=Path, default=DEFAULT_BGE)
    p.add_argument("--output-dir", type=Path, default=CANONICAL_ROOT / "artifacts" / "results" / "toolret_retrieval_heldout_e7_r1")
    p.add_argument("--allow-dirty", action="store_true")
    return p.parse_args()


def parse_csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_doc(raw: Any) -> dict[str, Any]:
    if isinstance(raw, str):
        return json.loads(raw)
    if isinstance(raw, dict):
        return raw
    raise TypeError(f"unsupported doc type: {type(raw)}")


def tool_id(doc: dict[str, Any]) -> str:
    return str(doc.get("id") or doc.get("name") or doc.get("api_name") or doc.get("path") or "")


def tool_category(doc: dict[str, Any]) -> str:
    return str(doc.get("category") or doc.get("category_name") or doc.get("domain") or "unknown")


def tool_text(tool: str, doc: dict[str, Any]) -> str:
    params = doc.get("parameters") or doc.get("required_parameters") or {}
    if isinstance(params, dict):
        ptxt = ", ".join(list(params.keys())[:12])
    elif isinstance(params, list):
        ptxt = ", ".join(str(p.get("name", "")) for p in params[:12] if isinstance(p, dict))
    else:
        ptxt = ""
    return "\n".join(
        [
            f"tool: {tool}",
            f"category: {tool_category(doc)}",
            f"description: {doc.get('description') or doc.get('api_description') or doc.get('documentation') or ''}",
            f"parameters: {ptxt}",
        ]
    )


def scenario_text(query: str, tool: str) -> str:
    return f"tool: {tool}\nwhen to use: {query}"


def norm_text(text: str) -> str:
    return " ".join(TOKEN_RE.findall(text.lower()))


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def cosine_near_duplicate(a: str, b: str) -> bool:
    # Cheap lexical proxy for split hygiene; dense ToolRet runs still use the requested retriever.
    ca = Counter(tokenize(a))
    cb = Counter(tokenize(b))
    if not ca or not cb:
        return False
    dot = sum(v * cb.get(k, 0) for k, v in ca.items())
    na = math.sqrt(sum(v * v for v in ca.values()))
    nb = math.sqrt(sum(v * v for v in cb.values()))
    return (dot / (na * nb)) >= 0.95


class BM25Index:
    def __init__(self, texts: list[str], *, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.term_counts = [Counter(tokenize(text)) for text in texts]
        self.doc_lens = np.asarray([sum(c.values()) for c in self.term_counts], dtype=np.float32)
        self.avgdl = float(self.doc_lens.mean()) if len(self.doc_lens) else 0.0
        df: Counter[str] = Counter()
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for counts in self.term_counts:
            df.update(counts.keys())
        for i, counts in enumerate(self.term_counts):
            for term, freq in counts.items():
                self.postings[term].append((i, freq))
        n_docs = max(1, len(self.term_counts))
        self.idf = {term: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}

    def scores(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.term_counts), dtype=np.float32)
        terms = set(tokenize(query))
        avgdl = self.avgdl or 1.0
        for term in terms:
            idf = self.idf.get(term, 0.0)
            if not idf:
                continue
            for i, freq in self.postings.get(term, []):
                dl = float(self.doc_lens[i]) or 1.0
                denom_base = self.k1 * (1.0 - self.b + self.b * dl / avgdl)
                scores[i] += idf * (freq * (self.k1 + 1.0)) / (freq + denom_base)
        return scores


def bge_model(path: Path, device: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(str(path), device=device)


def encode_bge(model: Any, texts: list[str], batch_size: int = 128) -> np.ndarray:
    return np.asarray(model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)


def normalize_matrix(m: np.ndarray) -> np.ndarray:
    if not m.size:
        return m
    norms = np.linalg.norm(m, axis=1)
    norms[norms == 0.0] = 1.0
    return m / norms[:, None]


def index_for(retriever: str, docs: list[dict[str, Any]], jina: JinaAIClient, bge: Any | None, embed_model: str) -> Any:
    texts = [d["text"] for d in docs]
    if retriever == "bm25":
        return BM25Index(texts)
    if retriever == "jina":
        return normalize_matrix(np.asarray(batched_query_embeddings(jina, texts, embed_model), dtype=np.float32))
    if retriever == "bge":
        if bge is None:
            raise RuntimeError("BGE requested but no model loaded")
        return encode_bge(bge, texts)
    raise ValueError(retriever)


def scores_for(retriever: str, index: Any, query: str, jina: JinaAIClient, bge: Any | None, embed_model: str) -> np.ndarray:
    if retriever == "bm25":
        return index.scores(query)
    if retriever == "jina":
        q = np.asarray(batched_query_embeddings(jina, [query], embed_model)[0], dtype=np.float32)
        q = q / (np.linalg.norm(q) or 1.0)
        return index @ q if getattr(index, "size", 0) else np.asarray([], dtype=np.float32)
    if retriever == "bge":
        q = encode_bge(bge, [query])[0]
        return index @ q if getattr(index, "size", 0) else np.asarray([], dtype=np.float32)
    raise ValueError(retriever)


def top_distinct(docs: list[dict[str, Any]], scores: np.ndarray, k: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for i in np.argsort(-scores):
        tool = docs[int(i)]["tool"]
        if tool in seen:
            continue
        seen.add(tool)
        out.append(tool)
        if len(out) >= k:
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
        if not s:
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
        p = min(1.0, 2.0 * sum(math.comb(n, k) for k in range(min(docs_only, shared_only) + 1)) / (2**n))
        method = "exact_binomial"
    else:
        stat = (abs(docs_only - shared_only) - 1.0) ** 2 / n
        p = math.erfc(math.sqrt(stat / 2.0))
        method = "mcnemar_chi2_cc"
    return {"compared": compared, "docs_only": docs_only, "shared_only": shared_only, "p_value": p, "method": method}


def run_arm(retriever: str, arm: str, docs: list[dict[str, Any]], eval_rows: list[dict[str, Any]], jina: JinaAIClient, bge: Any | None, embed_model: str, k: int, progress: ProgressLogger, context: dict[str, Any]) -> dict[str, Any]:
    idx = index_for(retriever, docs, jina, bge, embed_model)
    rows: list[dict[str, Any]] = []
    for n, item in enumerate(eval_rows, 1):
        cand = top_distinct(docs, scores_for(retriever, idx, item["query"], jina, bge, embed_model), k)
        gold = set(item["gold_tools"])
        rows.append(
            {
                "query_id": item["query_id"],
                "query": item["query"],
                "gold_tools": item["gold_tools"],
                "candidate_ids": cand,
                "top1": cand[0] if cand else None,
                "subset": item["subset"],
                "gold_in_top_5": bool(gold.intersection(cand[:5])),
                "top1_correct": bool(cand and cand[0] in gold),
            }
        )
        if n % 10000 == 0:
            progress.log("query_progress", retriever=retriever, arm=arm, processed=n, **context)
    by_subset = {s: summarize([r for r in rows if r["subset"] == s]) for s in ("heldout", "labeled")}
    out = {
        "paper_eligible": True,
        "config": {"retriever": retriever, "arm": arm, **context},
        "rows": rows,
        "metrics": {"overall": summarize(rows), **by_subset},
    }
    return out


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress = ProgressLogger(args.output_dir)
    t0 = time.perf_counter()
    commit_info = assert_clean_tree(allow_dirty=args.allow_dirty)
    commit = commit_info[0] if isinstance(commit_info, tuple) else commit_info
    progress.log("runner_start", repo_commit=commit, dataset="mangopy/ToolRet-Training-20w")
    embedder = resolve_local_embedder()
    progress.log("embedder_ready", **embedder)
    ds = load_dataset("mangopy/ToolRet-Training-20w", "ToolRet-Training-20w", split="train")
    by_cat_tool: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    queries_by_cat_tool: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in ds:
        docs = [parse_doc(x) for x in row["positive"]]
        seen_tools = set()
        for doc in docs:
            tid = tool_id(doc)
            if not tid or tid in seen_tools:
                continue
            seen_tools.add(tid)
            c = tool_category(doc)
            by_cat_tool[c].setdefault(tid, doc)
            queries_by_cat_tool[c][tid].append({"query_id": row["id"], "query": row["query"], "gold_tools": [tid]})
    retrievers = parse_csv(args.retrievers)
    categories = parse_csv(args.categories)
    seeds = [int(x) for x in parse_csv(args.seeds)]
    jina = JinaAIClient([])
    bge = None
    if "bge" in retrievers:
        device = os.environ.get("BGE_DEVICE") or ("cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
        progress.log("load_bge_begin", model_path=str(args.bge_model_path), device=device)
        bge = bge_model(args.bge_model_path, device)
        progress.log("load_bge_done", model_path=str(args.bge_model_path))
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config["repo_commit"] = commit
    summary: dict[str, Any] = {"paper_eligible": True, "config": config, "aggregate": {}}
    for category in categories:
        tools = by_cat_tool.get(category, {})
        eligible = [t for t, qs in queries_by_cat_tool[category].items() if len(qs) >= args.min_queries_per_tool]
        if len(tools) < args.min_tools or len(eligible) < args.min_tools:
            progress.log("category_skipped", category=category, total_tools=len(tools), eligible_tools=len(eligible), min_tools=args.min_tools)
            continue
        summary["aggregate"].setdefault(category, {})
        for seed in seeds:
            rng = random.Random(stable_hash({"category": category, "seed": seed}))
            heldout = set(rng.sample(sorted(eligible), int(round(len(eligible) * args.holdout_fraction))))
            eval_rows: list[dict[str, Any]] = []
            scenario_rows: list[dict[str, Any]] = []
            docs = [{"doc_id": stable_hash({"doc": category, "tool": t}), "tool": t, "kind": "description", "text": tool_text(t, doc)} for t, doc in sorted(tools.items())]
            for tool in sorted(eligible):
                qs = list(queries_by_cat_tool[category][tool])
                rng.shuffle(qs)
                cut = max(1, int(round(len(qs) * args.scenario_fraction)))
                scenario_qs = qs[:cut] if tool not in heldout else []
                eval_qs = qs[cut:] or qs[-1:]
                scenario_norms = [norm_text(q["query"]) for q in scenario_qs]
                for q in eval_qs:
                    if any(norm_text(q["query"]) == s or cosine_near_duplicate(q["query"], s) for s in scenario_norms):
                        continue
                    eval_rows.append({"query_id": f"{category}:{tool}:{q['query_id']}", "query": q["query"], "gold_tools": [tool], "subset": "heldout" if tool in heldout else "labeled"})
                for q in scenario_qs:
                    scenario_rows.append({"doc_id": stable_hash({"scenario": category, "tool": tool, "query": q["query_id"]}), "tool": tool, "kind": "scenario", "text": scenario_text(q["query"], tool)})
            shared_docs = docs + scenario_rows
            ctx = {
                "category": category,
                "seed": seed,
                "tool_count": len(tools),
                "eligible_tool_count": len(eligible),
                "heldout_tool_count": len(heldout),
                "heldout_tools_sha256": stable_hash(sorted(heldout)),
                "eval_query_count": len(eval_rows),
                "heldout_eval_query_count": sum(r["subset"] == "heldout" for r in eval_rows),
                "labeled_eval_query_count": sum(r["subset"] == "labeled" for r in eval_rows),
                "docs_doc_count": len(docs),
                "shared_doc_count": len(shared_docs),
                "split_rule": "per-tool query split; held-out tools contribute no scenario docs; exact and lexical-cos>=0.95 eval/scenario overlaps removed within tool",
            }
            progress.log("seed_begin", **ctx)
            for retriever in retrievers:
                rdir = args.output_dir / category / f"seed_{seed}" / retriever
                rdir.mkdir(parents=True, exist_ok=True)
                docs_out = run_arm(retriever, "docs_only", docs, eval_rows, jina, bge, args.embed_model, args.top_k, progress, ctx)
                shared_out = run_arm(retriever, "shared", shared_docs, eval_rows, jina, bge, args.embed_model, args.top_k, progress, ctx)
                docs_path = rdir / "docs_only.json"
                shared_path = rdir / "shared.json"
                save_json(docs_path, docs_out)
                save_json(shared_path, shared_out)
                m = mcnemar([r for r in docs_out["rows"] if r["subset"] == "heldout"], [r for r in shared_out["rows"] if r["subset"] == "heldout"])
                rsum = {
                    "docs_only": docs_out["metrics"],
                    "shared": shared_out["metrics"],
                    "shared_minus_docs_heldout_recall": shared_out["metrics"]["heldout"]["recall_at_5"] - docs_out["metrics"]["heldout"]["recall_at_5"],
                    "shared_vs_docs_mcnemar_heldout_recall": m,
                }
                save_json(rdir / "summary.json", rsum)
                summary["aggregate"][category].setdefault(str(seed), {})[retriever] = rsum
                progress.log("retriever_done", category=category, seed=seed, retriever=retriever, docs_heldout_recall=docs_out["metrics"]["heldout"]["recall_at_5"], shared_heldout_recall=shared_out["metrics"]["heldout"]["recall_at_5"], elapsed_seconds=round(time.perf_counter() - t0, 3))
    save_json(args.output_dir / "summary.json", summary)
    progress.log("complete", elapsed_seconds=round(time.perf_counter() - t0, 3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
