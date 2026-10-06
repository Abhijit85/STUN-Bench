#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_ROOT = Path(os.environ.get("FEDRAG_CANONICAL_ROOT", REPO_ROOT)).resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from math_qa import JinaAIClient
from scripts.run_stabletoolbench_federated import ProgressLogger, assert_clean_tree, resolve_local_embedder, save_json, stable_hash
from scripts.run_toolret_retrieval_heldout_e7 import (
    DEFAULT_BGE,
    bge_model,
    cosine_near_duplicate,
    index_for,
    mcnemar,
    parse_csv,
    run_arm,
    tool_text,
)


DROP_CONFIGS = {"apigen", "toolbench", "toolbench-sam"}
DEFAULT_CONFIGS = (
    "apibank,appbench,autotools-food,autotools-music,autotools-weather,"
    "craft-math-algebra,craft-tabmwp,craft-vqa,gorilla-huggingface,"
    "gorilla-pytorch,gorilla-tensor,gpt4tools,gta,metatool,mnms,"
    "restgpt-spotify,restgpt-tmdb,reversechain,rotbench,t-eval-dialog,"
    "t-eval-step,taskbench-daily,taskbench-huggingface,taskbench-multimedia,"
    "tool-be-honest,toolace,toolalpaca,toolemu,tooleyes,toolink,toollens,ultratool"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ToolRet sparse-config held-out retrieval-only replication.")
    p.add_argument("--configs", default=DEFAULT_CONFIGS)
    p.add_argument("--seeds", default="42,123,456")
    p.add_argument("--retrievers", default="bm25,jina,bge")
    p.add_argument("--holdout-fraction", type=float, default=0.30)
    p.add_argument("--min-queries-per-tool", type=int, default=2)
    p.add_argument("--scenario-count", type=int, default=1)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--near-duplicate-threshold", type=float, default=0.95)
    p.add_argument("--embed-model", default="jina-embeddings-v2-base-en")
    p.add_argument("--bge-model-path", type=Path, default=DEFAULT_BGE)
    p.add_argument("--output-dir", type=Path, default=CANONICAL_ROOT / "artifacts" / "results" / "toolret_sparse_heldout_e7_r1")
    p.add_argument("--allow-dirty", action="store_true")
    return p.parse_args()


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


def scenario_text(query: str, tool: str, source_config: str) -> str:
    return f"tool: {tool}\nsource: {source_config}\nwhen to use: {query}"


def load_sparse_pool(configs: list[str]) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    docs: dict[str, dict[str, Any]] = {}
    queries_by_tool: dict[str, list[dict[str, Any]]] = defaultdict(list)
    config_stats: dict[str, Any] = {}
    for cfg in configs:
        if cfg in DROP_CONFIGS:
            raise ValueError(f"refusing dropped config {cfg}")
        ds = load_dataset("mangopy/ToolRet-Queries", cfg, split="queries")
        tool_counts: Counter[str] = Counter()
        for row in ds:
            labels = parse_labels(row.get("labels"))
            gold: list[str] = []
            for label in labels:
                tool = label_tool_id(label)
                if not tool:
                    continue
                docs.setdefault(tool, label_doc(label))
                gold.append(tool)
            for tool in sorted(set(gold)):
                tool_counts[tool] += 1
                queries_by_tool[tool].append(
                    {
                        "query_id": str(row.get("id") or stable_hash({"cfg": cfg, "query": row.get("query")})),
                        "query": str(row.get("query") or row.get("instruction") or ""),
                        "instruction": str(row.get("instruction") or ""),
                        "gold_tools": [tool],
                        "source_config": cfg,
                    }
                )
        config_stats[cfg] = {
            "query_count": len(ds),
            "tool_count": len(tool_counts),
            "tools_ge_2": sum(v >= 2 for v in tool_counts.values()),
            "tools_ge_5": sum(v >= 5 for v in tool_counts.values()),
        }
    return docs, queries_by_tool, config_stats


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress = ProgressLogger(args.output_dir)
    t0 = time.perf_counter()
    commit_info = assert_clean_tree(allow_dirty=args.allow_dirty)
    commit = commit_info[0] if isinstance(commit_info, tuple) else commit_info
    configs = parse_csv(args.configs)
    dropped = sorted(set(configs).intersection(DROP_CONFIGS))
    if dropped:
        raise ValueError(f"refusing ToolBench/APIGen configs: {dropped}")
    progress.log("runner_start", repo_commit=commit, dataset="mangopy/ToolRet-Queries", configs=configs)
    progress.log("embedder_ready", **resolve_local_embedder())

    docs_by_tool, queries_by_tool, config_stats = load_sparse_pool(configs)
    eligible = sorted(t for t, qs in queries_by_tool.items() if len(qs) >= args.min_queries_per_tool)
    if not eligible:
        raise RuntimeError("no eligible tools after sparse ToolRet load")
    retrievers = parse_csv(args.retrievers)
    seeds = [int(x) for x in parse_csv(args.seeds)]
    jina = JinaAIClient([])
    bge = None
    if "bge" in retrievers:
        device = os.environ.get("BGE_DEVICE") or ("cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
        progress.log("load_bge_begin", model_path=str(args.bge_model_path), device=device)
        bge = bge_model(args.bge_model_path, device)
        progress.log("load_bge_done", model_path=str(args.bge_model_path))

    doc_rows = [
        {"doc_id": stable_hash({"doc": tool, "source": "toolret_sparse"}), "tool": tool, "kind": "description", "text": tool_text(tool, docs_by_tool[tool])}
        for tool in sorted(docs_by_tool)
    ]
    summary: dict[str, Any] = {
        "paper_eligible": True,
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()} | {
            "repo_commit": commit,
            "dropped_configs": sorted(DROP_CONFIGS),
            "source_configs": configs,
            "split_rule": "pooled sparse ToolRet-Queries configs; 30% of >=2-query tools held out; one labeled query per non-heldout tool becomes scenario; heldout tools contribute no scenario docs; exact and lexical-cos>=0.95 scenario/eval overlaps removed within tool",
        },
        "config_stats": config_stats,
        "tool_count": len(docs_by_tool),
        "eligible_tool_count": len(eligible),
        "aggregate": {},
    }

    for seed in seeds:
        rng = random.Random(stable_hash({"toolret_sparse": seed, "eligible": eligible}))
        heldout = set(rng.sample(eligible, int(round(len(eligible) * args.holdout_fraction))))
        eval_rows: list[dict[str, Any]] = []
        scenario_rows: list[dict[str, Any]] = []
        for tool in eligible:
            qs = list(queries_by_tool[tool])
            rng.shuffle(qs)
            if tool in heldout:
                scenario_qs: list[dict[str, Any]] = []
                eval_qs = qs
            else:
                scenario_qs = qs[: args.scenario_count]
                eval_qs = qs[args.scenario_count :]
            scenario_texts = [q["query"] for q in scenario_qs]
            for q in eval_qs:
                if any(q["query"].strip() == s.strip() or cosine_near_duplicate(q["query"], s) for s in scenario_texts):
                    continue
                eval_rows.append(
                    {
                        "query_id": f"{q['source_config']}:{tool}:{q['query_id']}",
                        "query": q["query"],
                        "gold_tools": [tool],
                        "subset": "heldout" if tool in heldout else "labeled",
                        "source_config": q["source_config"],
                    }
                )
            for q in scenario_qs:
                scenario_rows.append(
                    {
                        "doc_id": stable_hash({"scenario": q["source_config"], "tool": tool, "query": q["query_id"]}),
                        "tool": tool,
                        "kind": "scenario",
                        "source_config": q["source_config"],
                        "text": scenario_text(q["query"], tool, q["source_config"]),
                    }
                )
        shared_docs = doc_rows + scenario_rows
        ctx = {
            "seed": seed,
            "tool_count": len(docs_by_tool),
            "eligible_tool_count": len(eligible),
            "heldout_tool_count": len(heldout),
            "heldout_tools_sha256": stable_hash(sorted(heldout)),
            "eval_query_count": len(eval_rows),
            "heldout_eval_query_count": sum(r["subset"] == "heldout" for r in eval_rows),
            "labeled_eval_query_count": sum(r["subset"] == "labeled" for r in eval_rows),
            "docs_doc_count": len(doc_rows),
            "shared_doc_count": len(shared_docs),
            "source_config_count": len(configs),
        }
        progress.log("seed_begin", **ctx)
        summary["aggregate"].setdefault(str(seed), {})
        for retriever in retrievers:
            rdir = args.output_dir / f"seed_{seed}" / retriever
            rdir.mkdir(parents=True, exist_ok=True)
            docs_out = run_arm(retriever, "docs_only", doc_rows, eval_rows, jina, bge, args.embed_model, args.top_k, progress, ctx)
            shared_out = run_arm(retriever, "shared", shared_docs, eval_rows, jina, bge, args.embed_model, args.top_k, progress, ctx)
            for out in (docs_out, shared_out):
                out["config"]["source_configs"] = configs
                out["config"]["dropped_configs"] = sorted(DROP_CONFIGS)
            save_json(rdir / "docs_only.json", docs_out)
            save_json(rdir / "shared.json", shared_out)
            test = mcnemar([r for r in docs_out["rows"] if r["subset"] == "heldout"], [r for r in shared_out["rows"] if r["subset"] == "heldout"])
            rsum = {
                "docs_only": docs_out["metrics"],
                "shared": shared_out["metrics"],
                "shared_minus_docs_heldout_recall": shared_out["metrics"]["heldout"]["recall_at_5"] - docs_out["metrics"]["heldout"]["recall_at_5"],
                "shared_vs_docs_mcnemar_heldout_recall": test,
            }
            save_json(rdir / "summary.json", rsum)
            summary["aggregate"][str(seed)][retriever] = rsum
            progress.log(
                "retriever_done",
                seed=seed,
                retriever=retriever,
                docs_heldout_recall=docs_out["metrics"]["heldout"]["recall_at_5"],
                shared_heldout_recall=shared_out["metrics"]["heldout"]["recall_at_5"],
                docs_labeled_recall=docs_out["metrics"]["labeled"]["recall_at_5"],
                shared_labeled_recall=shared_out["metrics"]["labeled"]["recall_at_5"],
                elapsed_seconds=round(time.perf_counter() - t0, 3),
            )
    save_json(args.output_dir / "summary.json", summary)
    progress.log("complete", elapsed_seconds=round(time.perf_counter() - t0, 3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
