#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import json
import os
import statistics
import sys
import time
import types
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_ROOT = Path(os.environ.get("FEDRAG_CANONICAL_ROOT", "<REPO_ROOT>")).resolve()

DEFAULT_MODEL = CANONICAL_ROOT / "external_models" / "ToolBench" / "ToolBench_IR_bert_based_uncased"
DEFAULT_SOURCE = CANONICAL_ROOT / "artifacts" / "results" / "toolret_sparse_heldout_e7_r1"
DEFAULT_OUTPUT = CANONICAL_ROOT / "artifacts" / "results" / "toolret_sparse_toolbench_ir_e7_r1"


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def current_commit() -> str:
    import subprocess

    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def assert_clean_tree(allow_dirty: bool) -> str:
    import subprocess

    dirty = subprocess.check_output(["git", "status", "--short"], cwd=REPO_ROOT, text=True).strip()
    if dirty and not allow_dirty:
        raise RuntimeError(f"dirty worktree refused:\n{dirty}")
    return current_commit()


class ProgressLogger:
    def __init__(self, output_dir: Path) -> None:
        self.path = output_dir / "progress.jsonl"
        output_dir.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, **fields: Any) -> None:
        row = {"event": event, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **fields}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True), flush=True)


def stub_vision_audio() -> None:
    class _Interp:
        NEAREST = 0
        NEAREST_EXACT = 0
        BILINEAR = 2
        BICUBIC = 3
        BOX = 4
        HAMMING = 5
        LANCZOS = 1

    for name in [
        "torchvision",
        "torchvision.io",
        "torchvision.transforms",
        "torchvision.transforms.functional",
        "torchvision.transforms.v2",
        "torchvision.transforms.v2.functional",
        "torchaudio",
        "torchaudio._extension",
    ]:
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
            sys.modules[name] = m
    sys.modules["torchvision"].io = sys.modules["torchvision.io"]
    sys.modules["torchvision"].transforms = sys.modules["torchvision.transforms"]
    sys.modules["torchvision.io"].ImageReadMode = object
    sys.modules["torchvision.io"].decode_image = lambda *a, **k: None
    sys.modules["torchvision.transforms"].InterpolationMode = _Interp
    for fn in ["pil_to_tensor", "to_pil_image", "resize", "center_crop", "convert_image_dtype", "normalize"]:
        setattr(sys.modules["torchvision.transforms.functional"], fn, lambda *a, **k: None)


class ToolBenchIREncoder:
    def __init__(self, model_path: Path, device: str, batch_size: int) -> None:
        stub_vision_audio()
        from transformers import AutoModel, AutoTokenizer

        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        self.model = AutoModel.from_pretrained(str(model_path), local_files_only=True).to(device)
        self.model.eval()

    @torch.inference_mode()
    def encode(self, texts: list[str]) -> np.ndarray:
        outs: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            encoded = self.tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            output = self.model(**encoded)
            token_embeddings = output.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            outs.append(pooled.detach().cpu().numpy().astype("float32"))
        return np.concatenate(outs, axis=0) if outs else np.zeros((0, 768), dtype="float32")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay sparse ToolRet E7 with the released ToolBench IR retriever.")
    p.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--seeds", default="42,123,456")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--allow-dirty", action="store_true")
    return p.parse_args()


def parse_doc(raw: Any) -> dict[str, Any]:
    if isinstance(raw, str):
        return json.loads(raw)
    if isinstance(raw, dict):
        return raw
    raise TypeError(type(raw))


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


def load_toolret_queries(configs: list[str]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, str]]]:
    tools: dict[str, dict[str, Any]] = {}
    queries: dict[str, dict[str, str]] = {}
    for cfg in configs:
        ds = load_dataset("mangopy/ToolRet-Queries", cfg, split="queries")
        for row in ds:
            labels = row.get("labels")
            docs = json.loads(labels) if isinstance(labels, str) else labels
            if isinstance(docs, dict):
                docs = [docs]
            for raw in docs or []:
                doc = parse_doc(raw)
                tid = tool_id(doc)
                if not tid:
                    continue
                tools.setdefault(tid, doc)
                qid = f"{cfg}:{tid}:{row['id']}"
                queries[qid] = {"query_id": qid, "query": row["query"], "tool": tid}
    return tools, queries


def docs_for_tools(tools: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {"doc_id": stable_hash({"doc": tool}), "tool": tool, "kind": "description", "text": tool_text(tool, doc)}
        for tool, doc in sorted(tools.items())
    ]


def load_eval_rows(source_dir: Path, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = source_dir / f"seed_{seed}" / "jina" / "docs_only.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [
        {
            "query_id": r["query_id"],
            "query": r["query"],
            "gold_tools": r["gold_tools"],
            "subset": r["subset"],
        }
        for r in payload["rows"]
    ]
    return rows, payload["config"]


def shared_docs_for_seed(
    docs: list[dict[str, str]],
    queries_by_id: dict[str, dict[str, str]],
    eval_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    eval_ids = {r["query_id"] for r in eval_rows}
    heldout_tools = {r["gold_tools"][0] for r in eval_rows if r["subset"] == "heldout"}
    queries_by_tool: dict[str, list[dict[str, str]]] = defaultdict(list)
    for qid, q in sorted(queries_by_id.items()):
        queries_by_tool[q["tool"]].append(q)
    eligible_tools = {tool for tool, qs in queries_by_tool.items() if len(qs) >= 2}
    labeled_tools = eligible_tools - heldout_tools
    missing_by_tool: dict[str, list[dict[str, str]]] = defaultdict(list)
    for qid, q in sorted(queries_by_id.items()):
        tool = q["tool"]
        if tool in labeled_tools and qid not in eval_ids:
            missing_by_tool[tool].append(q)
    scenario_docs: list[dict[str, str]] = []
    for tool in sorted(labeled_tools):
        choices = missing_by_tool.get(tool, [])
        if not choices:
            raise RuntimeError(f"no scenario candidate for labeled tool {tool}")
        # The sparse E7 source run used one scenario query per non-heldout eligible tool.
        # Some tools have no surviving labeled eval rows after overlap filtering but still contribute a scenario.
        q = choices[0]
        scenario_docs.append(
            {
                "doc_id": stable_hash({"scenario": tool, "query_id": q["query_id"]}),
                "tool": tool,
                "kind": "scenario",
                "text": scenario_text(q["query"], tool),
            }
        )
    return docs + scenario_docs


def select_distinct(docs: list[dict[str, str]], scores: np.ndarray, top_k: int) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    for idx in np.argsort(-scores):
        tool = docs[int(idx)]["tool"]
        if tool in seen:
            continue
        seen.add(tool)
        selected.append(tool)
        if len(selected) >= top_k:
            break
    return selected


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def one(bucket: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "n": len(bucket),
            "recall_at_5": sum(bool(r["gold_in_top_5"]) for r in bucket) / len(bucket) if bucket else 0.0,
            "top1_accuracy": sum(bool(r["top1_correct"]) for r in bucket) / len(bucket) if bucket else 0.0,
            "shortfall_count": sum(len(set(r["candidate_ids"])) < 5 for r in bucket),
        }

    return {
        "overall": one(rows),
        "heldout": one([r for r in rows if r["subset"] == "heldout"]),
        "labeled": one([r for r in rows if r["subset"] == "labeled"]),
    }


def mcnemar(rows_docs: list[dict[str, Any]], rows_shared: list[dict[str, Any]]) -> dict[str, Any]:
    import math

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


def run_arm(arm: str, docs: list[dict[str, str]], eval_rows: list[dict[str, Any]], encoder: ToolBenchIREncoder, top_k: int) -> dict[str, Any]:
    doc_emb = encoder.encode([d["text"] for d in docs])
    query_emb = encoder.encode([r["query"] for r in eval_rows])
    rows: list[dict[str, Any]] = []
    for r, emb in zip(eval_rows, query_emb):
        scores = doc_emb @ emb if len(doc_emb) else np.asarray([], dtype="float32")
        cand = select_distinct(docs, scores, top_k)
        gold = set(r["gold_tools"])
        rows.append(
            {
                **r,
                "candidate_ids": cand,
                "top1": cand[0] if cand else None,
                "gold_in_top_5": bool(gold.intersection(cand[:5])),
                "top1_correct": bool(cand and cand[0] in gold),
            }
        )
    return {"paper_eligible": True, "config": {"arm": arm}, "rows": rows, "metrics": summarize(rows)}


def aggregate(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values) if values else 0.0,
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress = ProgressLogger(args.output_dir)
    commit = assert_clean_tree(args.allow_dirty)
    source_summary = json.loads((args.source_dir / "summary.json").read_text(encoding="utf-8"))
    configs = source_summary["config"]["source_configs"]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    progress.log("runner_start", repo_commit=commit, source_dir=str(args.source_dir), output_dir=str(args.output_dir))
    tools, queries_by_id = load_toolret_queries(configs)
    docs = docs_for_tools(tools)
    if len(docs) != int(source_summary["tool_count"]):
        raise RuntimeError(f"doc count mismatch: {len(docs)} != {source_summary['tool_count']}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = ToolBenchIREncoder(args.model_path, device=device, batch_size=args.batch_size)
    model_info = {
        "checkpoint_id": "ToolBench/ToolBench_IR_bert_based_uncased",
        "resolved_revision": "cf4a904",
        "model_path": str(args.model_path),
        "config_sha256": sha256_file(args.model_path / "config.json"),
        "model_sha256": sha256_file(args.model_path / "pytorch_model.bin"),
    }
    summary: dict[str, Any] = {
        "paper_eligible": True,
        "config": {
            "repo_commit": commit,
            "source_dir": str(args.source_dir),
            "source_summary_sha256": sha256_file(args.source_dir / "summary.json"),
            "source_configs": configs,
            "seeds": seeds,
            "top_k": args.top_k,
            "candidate_rule": "distinct5_walkdown",
            "model": model_info,
        },
        "aggregate": {},
    }
    per_seed: dict[str, Any] = {}
    for seed in seeds:
        eval_rows, source_config = load_eval_rows(args.source_dir, seed)
        shared_docs = shared_docs_for_seed(docs, queries_by_id, eval_rows)
        if len(shared_docs) != int(source_config["shared_doc_count"]):
            raise RuntimeError(f"seed {seed} shared doc count mismatch: {len(shared_docs)} != {source_config['shared_doc_count']}")
        heldout_tools = sorted({r["gold_tools"][0] for r in eval_rows if r["subset"] == "heldout"})
        visible_heldout_sha = stable_hash(heldout_tools)
        if len(heldout_tools) != int(source_config["heldout_tool_count"]):
            raise RuntimeError(f"seed {seed} visible heldout count mismatch: {len(heldout_tools)} != {source_config['heldout_tool_count']}")
        if len(eval_rows) != int(source_config["eval_query_count"]):
            raise RuntimeError(f"seed {seed} eval count mismatch")
        context = {
            "seed": seed,
            "eval_query_count": len(eval_rows),
            "heldout_eval_query_count": sum(r["subset"] == "heldout" for r in eval_rows),
            "labeled_eval_query_count": sum(r["subset"] == "labeled" for r in eval_rows),
            "docs_doc_count": len(docs),
            "shared_doc_count": len(shared_docs),
            "heldout_tools_sha256": source_config["heldout_tools_sha256"],
            "visible_heldout_tools_sha256": visible_heldout_sha,
        }
        progress.log("seed_begin", **context)
        sdir = args.output_dir / f"seed_{seed}" / "toolbench_ir"
        docs_out = run_arm("docs_only", docs, eval_rows, encoder, args.top_k)
        shared_out = run_arm("shared", shared_docs, eval_rows, encoder, args.top_k)
        for payload, arm in ((docs_out, "docs_only"), (shared_out, "shared")):
            payload["config"].update({"retriever": "toolbench_ir", **context, "model": model_info})
            save_json(sdir / f"{arm}.json", payload)
        m = mcnemar(
            [r for r in docs_out["rows"] if r["subset"] == "heldout"],
            [r for r in shared_out["rows"] if r["subset"] == "heldout"],
        )
        seed_summary = {
            "docs_only": docs_out["metrics"],
            "shared": shared_out["metrics"],
            "shared_minus_docs_heldout_recall": shared_out["metrics"]["heldout"]["recall_at_5"] - docs_out["metrics"]["heldout"]["recall_at_5"],
            "shared_vs_docs_mcnemar_heldout_recall": m,
            **context,
        }
        save_json(sdir / "summary.json", seed_summary)
        per_seed[str(seed)] = seed_summary
        progress.log(
            "seed_done",
            seed=seed,
            docs_heldout=docs_out["metrics"]["heldout"]["recall_at_5"],
            shared_heldout=shared_out["metrics"]["heldout"]["recall_at_5"],
            docs_labeled=docs_out["metrics"]["labeled"]["recall_at_5"],
            shared_labeled=shared_out["metrics"]["labeled"]["recall_at_5"],
        )
    summary["seeds"] = per_seed
    for arm in ("docs_only", "shared"):
        summary["aggregate"][arm] = {
            subset: {
                metric: aggregate([per_seed[str(seed)][arm][subset][metric] for seed in seeds])
                for metric in ("recall_at_5", "top1_accuracy")
            }
            for subset in ("overall", "heldout", "labeled")
        }
    save_json(args.output_dir / "summary.json", summary)
    progress.log("complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
