#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path
from collections import defaultdict
from typing import Any

import numpy as np
from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_ROOT = Path(os.environ.get("FEDRAG_CANONICAL_ROOT", REPO_ROOT)).resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from math_qa import JinaAIClient
from scripts.run_gsm8k_small_router_sweep import _load_local_backend
from scripts.run_stabletoolbench_federated import (
    ProgressLogger,
    assert_clean_tree,
    batched_query_embeddings,
    resolve_local_embedder,
    save_json,
    stable_hash,
    temporary_env,
)
from scripts.run_stabletoolbench_heldout_retriever_compare import (
    BM25Index,
    bge_encode,
    build_bge_encoder,
    sha256_file,
)
from scripts.run_stabletoolbench_symmetric_expansion_d3 import (
    GEN_PROMPT_TEMPLATE,
    local_generate_queries,
    prompt_hash,
)

DEFAULT_SOURCE = CANONICAL_ROOT / "artifacts" / "results" / "toolret_sparse_heldout_e7_r1"
DEFAULT_OUTPUT = CANONICAL_ROOT / "artifacts" / "results" / "toolret_sparse_symmetric_expansion_e1_r1"
DEFAULT_GENERATOR = "<HF_CACHE>/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
DEFAULT_BGE = Path("<HF_CACHE>/models--BAAI--bge-base-en-v1.5/snapshots/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a")
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def scenario_text(query: str, tool: str, source_config: str) -> str:
    return f"tool: {tool}\nsource: {source_config}\nwhen to use: {query}"


def norm_text(text: str) -> str:
    return " ".join(TOKEN_RE.findall(text.lower()))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E1 independent-pool LLM symmetric expansion on sparse ToolRet split.")
    p.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--generated-query-file", type=Path, default=None)
    p.add_argument("--generator-model-path", default=DEFAULT_GENERATOR)
    p.add_argument("--generator-temperature", type=float, default=0.7)
    p.add_argument("--generator-max-new-tokens", type=int, default=192)
    p.add_argument("--synthetic-per-tool", type=int, default=5)
    p.add_argument("--cap-per-tool", type=int, default=5)
    p.add_argument("--seeds", default="42,123,456")
    p.add_argument("--retrievers", default="jina,bm25,bge")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--embed-model", default="jina-embeddings-v2-base-en")
    p.add_argument("--bge-model-path", type=Path, default=DEFAULT_BGE)
    p.add_argument("--allow-dirty", action="store_true")
    return p.parse_args()


def parse_csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_doc(raw: Any) -> dict[str, Any]:
    if isinstance(raw, str):
        return json.loads(raw)
    if isinstance(raw, dict):
        return raw
    raise TypeError(f"unsupported doc type: {type(raw)}")


def parse_labels(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            return []
    else:
        value = raw
    return [x for x in value if isinstance(x, dict)] if isinstance(value, list) else []


def label_tool_id(label: dict[str, Any]) -> str:
    if label.get("id"):
        return str(label["id"])
    doc = label.get("doc") if isinstance(label.get("doc"), dict) else label
    for key in ("id", "name", "api_name", "tool_name", "doc_id"):
        if doc.get(key):
            return str(doc[key])
    return stable_hash(doc)[:16]


def label_doc(label: dict[str, Any]) -> dict[str, Any]:
    doc = label.get("doc") if isinstance(label.get("doc"), dict) else label
    return dict(doc)


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


def load_toolret_queries(configs: list[str]) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    tools: dict[str, dict[str, Any]] = {}
    queries: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cfg in configs:
        ds = load_dataset("mangopy/ToolRet-Queries", cfg, split="queries")
        for row in ds:
            labels = parse_labels(row.get("labels"))
            gold: list[str] = []
            for label in labels:
                tid = label_tool_id(label)
                if not tid:
                    continue
                tools.setdefault(tid, label_doc(label))
                gold.append(tid)
            for tid in sorted(set(gold)):
                queries[tid].append(
                    {
                        "query_id": str(row.get("id") or stable_hash({"cfg": cfg, "query": row.get("query")})),
                        "query": str(row.get("query") or row.get("instruction") or ""),
                        "instruction": str(row.get("instruction") or ""),
                        "gold_tools": [tid],
                        "source_config": cfg,
                    }
                )
    return tools, queries


def docs_for_tools(tools: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "doc_id": stable_hash({"doc": tool}),
            "tool": tool,
            "kind": "description",
            "text": tool_text(tool, doc),
        }
        for tool, doc in sorted(tools.items())
    ]


def generated_query_file(args: argparse.Namespace) -> Path:
    return args.generated_query_file or (args.output_dir / "generated_queries.json")


def load_eval_rows(source_dir: Path, seed: int) -> list[dict[str, Any]]:
    payload = json.loads((source_dir / f"seed_{seed}" / "jina" / "docs_only.json").read_text(encoding="utf-8"))
    return [
        {
            "query_id": row["query_id"],
            "query": row["query"],
            "gold_tools": list(row["gold_tools"]),
            "subset": row["subset"],
        }
        for row in payload["rows"]
    ]


def make_synthetic_records(docs: list[dict[str, str]], args: argparse.Namespace, progress: ProgressLogger) -> dict[str, Any]:
    path = generated_query_file(args)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        progress.log("generated_queries_loaded", path=str(path), sha256=sha256_file(path), complete=payload.get("complete"))
        return payload
    backend = _load_local_backend(str(args.generator_model_path), "auto")
    records: dict[str, Any] = {}
    progress.log("llm_generation_begin", tool_count=len(docs), synthetic_per_tool=args.synthetic_per_tool, model_path=str(args.generator_model_path))
    for idx, doc in enumerate(docs, 1):
        tool = doc["tool"]
        queries, raw = local_generate_queries(
            backend,
            tool=tool,
            description=doc["text"],
            count=args.synthetic_per_tool,
            temperature=args.generator_temperature,
            max_new_tokens=args.generator_max_new_tokens,
        )
        records[tool] = {"description": doc["text"], "queries": queries, "raw": raw}
        if idx % 100 == 0:
            payload = generated_payload(args, records, complete=False)
            save_json(path, payload)
            progress.log("llm_generation_progress", generated_tools=idx, total_tools=len(docs), path=str(path))
    payload = generated_payload(args, records, complete=True)
    save_json(path, payload)
    progress.log("llm_generation_done", generated_tools=len(records), path=str(path), sha256=sha256_file(path))
    return payload


def generated_payload(args: argparse.Namespace, records: dict[str, Any], *, complete: bool) -> dict[str, Any]:
    return {
        "generation_mode": "llm_from_description",
        "generator_model_path": str(args.generator_model_path),
        "generator_model_id": "Llama-3.1-8B-Instruct",
        "temperature": args.generator_temperature,
        "prompt_template": GEN_PROMPT_TEMPLATE,
        "prompt_sha256": prompt_hash(GEN_PROMPT_TEMPLATE),
        "synthetic_per_tool": args.synthetic_per_tool,
        "records": records,
        "complete": complete,
    }


def synthetic_docs(base_docs: list[dict[str, str]], payload: dict[str, Any], count: int) -> list[dict[str, str]]:
    records = payload.get("records") or {}
    out: list[dict[str, str]] = []
    for doc in base_docs:
        tool = doc["tool"]
        queries = list((records.get(tool) or {}).get("queries") or [])[:count]
        for idx, query in enumerate(queries):
            out.append({
                "doc_id": stable_hash({"synthetic": tool, "idx": idx, "query": query}),
                "tool": tool,
                "kind": "synthetic_query",
                "text": str(query),
            })
    return out


def scenario_docs_for_seed(queries_by_tool: dict[str, list[dict[str, Any]]], eval_rows: list[dict[str, Any]], *, seed: int) -> list[dict[str, str]]:
    eval_ids = {str(row["query_id"]).split(":")[-1] for row in eval_rows}
    heldout_tools = {row["gold_tools"][0] for row in eval_rows if row["subset"] == "heldout"}
    eligible = sorted({row["gold_tools"][0] for row in eval_rows})
    docs: list[dict[str, str]] = []
    # Reproduce the source split runner's per-seed shuffle order before it takes the first
    # non-heldout query as the scenario. The docs-only guard protects against any drift in
    # description indexing; this preserves shared-index provenance for the expansion arms.
    import random

    rng = random.Random(stable_hash({"toolret_sparse": seed, "eligible": eligible}))
    for tool in eligible:
        if tool in heldout_tools:
            continue
        choices = list(queries_by_tool[tool])
        rng.shuffle(choices)
        choices = [q for q in choices if q["query_id"] not in eval_ids]
        if not choices:
            continue
        q = choices[0]
        docs.append(
            {
                "doc_id": stable_hash({"scenario": q["source_config"], "tool": tool, "query": q["query_id"]}),
                "tool": tool,
                "kind": "scenario",
                "source_config": q["source_config"],
                "text": scenario_text(q["query"], tool, q["source_config"]),
            }
        )
    return docs


def capped_docs(synth: list[dict[str, str]], scenarios: list[dict[str, str]], cap: int) -> list[dict[str, str]]:
    by_synth: dict[str, list[dict[str, str]]] = {}
    by_exp: dict[str, list[dict[str, str]]] = {}
    for doc in synth:
        by_synth.setdefault(doc["tool"], []).append(doc)
    for doc in scenarios:
        by_exp.setdefault(doc["tool"], []).append(doc)
    out: list[dict[str, str]] = []
    for tool in sorted(by_synth):
        selected = list(by_exp.get(tool, [])[:cap])
        selected.extend(by_synth[tool][: max(0, cap - len(selected))])
        out.extend(selected[:cap])
    return out


def index_for(retriever: str, texts: list[str], jina: JinaAIClient, embed_model: str, bge_model: Any | None) -> Any:
    if retriever == "bm25":
        return BM25Index(texts)
    if retriever == "jina":
        matrix = np.asarray(batched_query_embeddings(jina, texts, embed_model), dtype=np.float32)
        if matrix.size:
            norms = np.linalg.norm(matrix, axis=1)
            norms[norms == 0.0] = 1.0
            matrix = matrix / norms[:, None]
        return matrix
    if retriever == "bge":
        if bge_model is None:
            raise RuntimeError("BGE requested without model")
        return bge_encode(bge_model, texts)
    raise ValueError(retriever)


def scores_for(retriever: str, index: Any, query: str, jina: JinaAIClient, embed_model: str, bge_model: Any | None) -> np.ndarray:
    if retriever == "bm25":
        return index.encode_query(query)
    if retriever == "jina":
        q = np.asarray(batched_query_embeddings(jina, [query], embed_model)[0], dtype=np.float32)
        q = q / (np.linalg.norm(q) or 1.0)
        return index @ q if getattr(index, "size", 0) else np.asarray([], dtype=np.float32)
    if retriever == "bge":
        q = bge_encode(bge_model, [query])[0]
        return index @ q if getattr(index, "size", 0) else np.asarray([], dtype=np.float32)
    raise ValueError(retriever)


def select_distinct(docs: list[dict[str, str]], scores: np.ndarray, k: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for idx in np.argsort(-scores):
        tool = docs[int(idx)]["tool"]
        if tool in seen:
            continue
        seen.add(tool)
        out.append(tool)
        if len(out) >= k:
            break
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "recall_at_5": 0.0, "top1_accuracy": 0.0, "mean_distinct_tools": 0.0, "lt5_share": 0.0}
    return {
        "n": len(rows),
        "recall_at_5": sum(bool(row["gold_in_top_5"]) for row in rows) / len(rows),
        "top1_accuracy": sum(bool(row["top1_correct"]) for row in rows) / len(rows),
        "mean_distinct_tools": sum(int(row["distinct_tool_count"]) for row in rows) / len(rows),
        "lt5_share": sum(int(row["distinct_tool_count"]) < 5 for row in rows) / len(rows),
    }


def evaluate(arm: str, retriever: str, docs: list[dict[str, str]], eval_rows: list[dict[str, Any]], jina: JinaAIClient, embed_model: str, bge_model: Any | None, top_k: int) -> dict[str, Any]:
    index = index_for(retriever, [doc["text"] for doc in docs], jina, embed_model, bge_model)
    rows: list[dict[str, Any]] = []
    for item in eval_rows:
        candidates = select_distinct(docs, scores_for(retriever, index, item["query"], jina, embed_model, bge_model), top_k)
        if len(candidates) != top_k:
            raise RuntimeError(
                f"{arm}/{retriever} query {item['query_id']} returned {len(candidates)} distinct tools, expected {top_k}"
            )
        gold = set(item["gold_tools"])
        rows.append({
            **item,
            "candidate_ids": candidates,
            "distinct_tool_count": len(candidates),
            "gold_in_top_5": bool(gold.intersection(candidates[:top_k])),
            "top1_correct": bool(candidates and candidates[0] in gold),
        })
    return {
        "paper_eligible": True,
        "config": {"arm": arm, "retriever": retriever, "doc_count": len(docs)},
        "rows": rows,
        "metrics": {
            "overall": summarize(rows),
            "heldout": summarize([row for row in rows if row["subset"] == "heldout"]),
            "labeled": summarize([row for row in rows if row["subset"] == "labeled"]),
        },
    }


def mean_sd(values: list[float]) -> dict[str, float]:
    return {"mean": statistics.mean(values), "sd": statistics.stdev(values) if len(values) > 1 else 0.0}


def leak_report(synth_docs: list[dict[str, str]], eval_rows: list[dict[str, Any]], bge_model: Any | None) -> dict[str, Any]:
    synth_norm = {norm_text(doc["text"]) for doc in synth_docs}
    eval_norm = {norm_text(row["query"]) for row in eval_rows}
    report = {
        "generated_query_count": len(synth_docs),
        "exact_matches_eval": len(synth_norm & eval_norm),
    }
    if bge_model is not None and synth_docs and eval_rows:
        synth_texts = [doc["text"] for doc in synth_docs]
        eval_texts = sorted({row["query"] for row in eval_rows})
        synth_emb = bge_encode(bge_model, synth_texts)
        eval_emb = bge_encode(bge_model, eval_texts)
        maxima: list[float] = []
        for start in range(0, synth_emb.shape[0], 4096):
            sims = synth_emb[start : start + 4096] @ eval_emb.T
            maxima.extend(np.max(sims, axis=1).tolist())
        report.update(
            {
                "near_dup_095_count": sum(score >= 0.95 for score in maxima),
                "near_dup_090_count": sum(score >= 0.90 for score in maxima),
                "max_similarity_max": max(maxima) if maxima else 0.0,
                "similarity_model": str(DEFAULT_BGE),
            }
        )
    return report


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress = ProgressLogger(args.output_dir)
    t0 = time.perf_counter()
    commit_info = assert_clean_tree(allow_dirty=args.allow_dirty)
    commit = commit_info[0] if isinstance(commit_info, tuple) else commit_info
    source_summary = json.loads((args.source_dir / "summary.json").read_text(encoding="utf-8"))
    configs = source_summary["config"]["source_configs"]
    seeds = [int(x) for x in parse_csv(args.seeds)]
    retrievers = parse_csv(args.retrievers)
    progress.log("runner_start", repo_commit=commit, source_dir=str(args.source_dir), configs=len(configs))
    tools, queries_by_tool = load_toolret_queries(configs)
    base_docs = docs_for_tools(tools)
    if len(base_docs) != int(source_summary["tool_count"]):
        raise RuntimeError(f"tool count mismatch: {len(base_docs)} != {source_summary['tool_count']}")
    payload = make_synthetic_records(base_docs, args, progress)
    synth = synthetic_docs(base_docs, payload, args.synthetic_per_tool)
    embedder = resolve_local_embedder()
    jina = JinaAIClient([])
    bge_model = None
    if "bge" in retrievers:
        device = os.environ.get("BGE_DEVICE") or ("cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
        progress.log("load_bge_begin", model_path=str(args.bge_model_path), device=device)
        bge_model = build_bge_encoder(args.bge_model_path, device=device)
        progress.log("load_bge_done", model_sha256=sha256_file(args.bge_model_path / "config.json"), device=device)
    embedder = resolve_local_embedder()
    all_eval = [row for seed in seeds for row in load_eval_rows(args.source_dir, seed)]
    exact_leak = leak_report(synth, all_eval, bge_model)
    summary: dict[str, Any] = {
        "paper_eligible": True,
        "repo_commit": commit,
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()} | {
            "source_summary_sha256": sha256_file(args.source_dir / "summary.json"),
            "source_configs": configs,
            "tool_count": len(base_docs),
            "generator_prompt_sha256": prompt_hash(GEN_PROMPT_TEMPLATE),
            "embedder": embedder,
        },
        "leak_report": exact_leak,
        "aggregate": {},
    }
    progress.log("leak_check_done", **exact_leak)
    local_jina_env = {
        "JINA_LOCAL_EMBED_MODEL": embedder["model_path"],
        "JINA_LOCAL_EMBED_DEVICE": embedder["device"],
        "JINA_LOCAL_EMBED_LOCAL_ONLY": embedder["local_only"],
        "JINA_API_KEY": None,
    }
    with temporary_env(local_jina_env):
        for seed in seeds:
            eval_rows = load_eval_rows(args.source_dir, seed)
            scenarios = scenario_docs_for_seed(queries_by_tool, eval_rows, seed=seed)
            arms = {
                "docs_only": base_docs,
                "synthetic_docs": base_docs + synth,
                "synthetic_plus_experience": base_docs + synth + scenarios,
                "capped_synthetic_plus_experience": base_docs + capped_docs(synth, scenarios, args.cap_per_tool),
            }
            for retriever in retrievers:
                for arm, docs in arms.items():
                    out = evaluate(arm, retriever, docs, eval_rows, jina, args.embed_model, bge_model, args.top_k)
                    if arm == "docs_only":
                        source_docs_path = args.source_dir / f"seed_{seed}" / retriever / "docs_only.json"
                        source_docs = json.loads(source_docs_path.read_text(encoding="utf-8"))
                        for subset in ("heldout", "labeled", "overall"):
                            got = float(out["metrics"][subset]["recall_at_5"])
                            want = float(source_docs["metrics"][subset]["recall_at_5"])
                            if abs(got - want) > 1e-12:
                                raise RuntimeError(
                                    f"docs_only mismatch seed={seed} retriever={retriever} subset={subset}: got {got} expected {want} from {source_docs_path}"
                                )
                        out["docs_only_source_path"] = str(source_docs_path)
                        out["docs_only_source_sha256"] = sha256_file(source_docs_path)
                    out["config"].update({"seed": seed, "source_dir": str(args.source_dir)})
                    path = args.output_dir / f"seed_{seed}" / retriever / f"{arm}.json"
                    save_json(path, out)
                    progress.log("arm_done", seed=seed, retriever=retriever, arm=arm, heldout_recall_at_5=out["metrics"]["heldout"]["recall_at_5"], labeled_recall_at_5=out["metrics"]["labeled"]["recall_at_5"], elapsed_seconds=round(time.perf_counter() - t0, 3))
    for retriever in retrievers:
        for arm in ("docs_only", "synthetic_docs", "synthetic_plus_experience", "capped_synthetic_plus_experience"):
            vals: dict[str, list[float]] = {"heldout_recall_at_5": [], "labeled_recall_at_5": [], "overall_recall_at_5": []}
            for seed in seeds:
                payload_out = json.loads((args.output_dir / f"seed_{seed}" / retriever / f"{arm}.json").read_text(encoding="utf-8"))
                vals["heldout_recall_at_5"].append(float(payload_out["metrics"]["heldout"]["recall_at_5"]))
                vals["labeled_recall_at_5"].append(float(payload_out["metrics"]["labeled"]["recall_at_5"]))
                vals["overall_recall_at_5"].append(float(payload_out["metrics"]["overall"]["recall_at_5"]))
            summary["aggregate"][f"{arm}/{retriever}"] = {metric: mean_sd(series) for metric, series in vals.items()}
    save_json(args.output_dir / "summary.json", summary)
    progress.log("complete", output_dir=str(args.output_dir), elapsed_seconds=round(time.perf_counter() - t0, 3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
