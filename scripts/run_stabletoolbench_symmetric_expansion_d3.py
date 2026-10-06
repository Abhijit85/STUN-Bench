#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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
    load_package_file,
    load_stabletoolbench_queries,
    package_to_candidates,
    resolve_local_embedder,
    save_json,
    stable_hash,
    temporary_env,
)
from scripts.run_stabletoolbench_heldout import HELDOUT_GROUPS, heldout_tool_set
from scripts.run_stabletoolbench_heldout_retriever_compare import (
    BM25Index,
    bge_encode,
    build_bge_encoder,
    sha256_file,
)
from scripts.run_gsm8k_small_router_sweep import _load_local_backend
from synapse.knowledge.compendium import KnowledgeArtifact, KnowledgePackage

DEFAULT_OUTPUT_DIR = CANONICAL_ROOT / "artifacts" / "results" / "stabletoolbench_symmetric_expansion_d3_r1"
DEFAULT_BGE = Path("<HF_CACHE>/models--BAAI--bge-base-en-v1.5/snapshots/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="D3: symmetric catalog-wide synthetic expansion retrieval control.")
    p.add_argument("--stb-root", type=Path, default=DEFAULT_STB_ROOT)
    p.add_argument("--groups", type=str, default=",".join(GROUPS))
    p.add_argument("--seeds", type=str, default="42,123,456")
    p.add_argument("--e6-dirs", type=str, required=True, help="Comma-separated seed=dir entries containing E6 packages.")
    p.add_argument("--retrievers", type=str, default="jina,bm25,bge")
    p.add_argument("--synthetic-per-tool", type=int, default=8)
    p.add_argument("--cap-per-tool", type=int, default=8)
    p.add_argument("--generation-mode", choices=["template_from_description", "llm_from_description"], default="template_from_description")
    p.add_argument("--generated-query-file", type=Path, default=None)
    p.add_argument("--generator-model-path", type=str, default=DEFAULT_MODEL_PATH)
    p.add_argument("--generator-temperature", type=float, default=0.7)
    p.add_argument("--generator-max-new-tokens", type=int, default=192)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--retrieval-pool-size", type=int, default=200)
    p.add_argument("--retrieval-mode", type=str, default="distinct_tool_topk")
    p.add_argument("--embed-model", type=str, default="jina-embeddings-v2-base-en")
    p.add_argument("--bge-model-path", type=Path, default=DEFAULT_BGE)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--allow-dirty", action="store_true")
    return p.parse_args()


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_seed_dirs(value: str) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for part in parse_csv(value):
        seed, path = part.split("=", 1)
        out[int(seed)] = Path(path)
    return out


def normalize_matrix(vectors: list[list[float]]) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.size:
        norms = np.linalg.norm(matrix, axis=1)
        norms[norms == 0.0] = 1.0
        matrix = matrix / norms[:, None]
    return matrix


def package_hash(package: KnowledgePackage) -> str:
    return stable_hash(
        [
            {
                "signature": artifact.signature,
                "text": artifact.text,
                "structured_payload": artifact.structured_payload,
                "metadata": artifact.metadata,
            }
            for artifact in sorted(package.artifacts, key=lambda a: a.signature)
        ]
    )


def tool_doc_artifacts(package: KnowledgePackage) -> list[KnowledgeArtifact]:
    docs = []
    for artifact in package.artifacts:
        payload = artifact.structured_payload or {}
        if artifact.metadata.get("artifact_origin") == "tool_doc" or payload.get("type") == "tool_doc":
            docs.append(artifact)
    return docs or list(package.artifacts)


def experience_artifacts(package: KnowledgePackage) -> list[KnowledgeArtifact]:
    artifacts = []
    for artifact in package.artifacts:
        payload = artifact.structured_payload or {}
        if artifact.metadata.get("artifact_origin") == "tool_doc" or payload.get("type") == "tool_doc":
            continue
        tool = str(artifact.metadata.get("tool") or "").strip()
        if tool:
            artifacts.append(artifact)
    return artifacts


def description_for(artifact: KnowledgeArtifact) -> tuple[str, dict[str, Any]]:
    payload = artifact.structured_payload or {}
    meta = artifact.metadata or {}
    tool = str(meta.get("tool") or payload.get("tool") or artifact.signature)
    desc = str(payload.get("tool_description") or payload.get("description") or artifact.text or "").strip()
    category = str(meta.get("category") or payload.get("category") or "").strip()
    aliases = payload.get("aliases") if isinstance(payload.get("aliases"), list) else []
    return tool, {"description": desc, "category": category, "aliases": [str(x) for x in aliases if str(x).strip()]}


def synthetic_texts(tool: str, info: dict[str, Any], count: int) -> list[str]:
    desc = info.get("description") or f"Use {tool} for its documented API tasks."
    category = info.get("category") or "tool"
    alias = ", ".join(info.get("aliases") or []) or tool
    templates = [
        "I need a tool for: {desc}",
        "Which API should handle this request: {desc}",
        "Find the {category} tool that can do this: {desc}",
        "Route a user request about {alias} capabilities: {desc}",
        "Use this tool when the task matches: {desc}",
        "Select the API for {category} work described as: {desc}",
        "A user asks for {alias}. Relevant documentation: {desc}",
        "Tool-routing query for {tool}: {desc}",
        "What tool supports the following need? {desc}",
        "Choose the best API for a request in {category}: {desc}",
        "The user needs functionality like {alias}; documentation says: {desc}",
        "Candidate intent for {tool}: {desc}",
    ]
    return [templates[i % len(templates)].format(tool=tool, desc=desc, category=category, alias=alias) for i in range(count)]


GEN_PROMPT_TEMPLATE = "Write five different user requests that this tool would be the right choice for. Tool: {tool}. Description: {description}."


def prompt_hash(prompt_template: str) -> str:
    return hashlib.sha256(prompt_template.encode("utf-8")).hexdigest()


def parse_generated_queries(text: str, count: int) -> list[str]:
    out: list[str] = []
    for line in text.splitlines():
        cleaned = line.strip()
        cleaned = re.sub(r"^[-*\u2022\s]+", "", cleaned)
        cleaned = re.sub(r"^\d+[.)]\s*", "", cleaned).strip()
        cleaned = cleaned.strip('"')
        if cleaned and cleaned.lower() not in {"sure", "here are five different user requests:"}:
            out.append(cleaned)
        if len(out) >= count:
            break
    return out


def local_generate_queries(backend: Any, *, tool: str, description: str, count: int, temperature: float, max_new_tokens: int) -> tuple[list[str], str]:
    import torch

    prompt = GEN_PROMPT_TEMPLATE.format(tool=tool, description=description or f"Use {tool} for its documented API tasks.")
    messages = [{"role": "user", "content": prompt}]
    rendered = backend.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = backend.tokenizer(rendered, return_tensors="pt")
    encoded = {key: value.to(backend.model.device) for key, value in encoded.items()}
    with torch.no_grad():
        output_ids = backend.model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            pad_token_id=backend.tokenizer.pad_token_id,
            eos_token_id=backend.tokenizer.eos_token_id,
        )
    prompt_len = encoded["input_ids"].shape[1]
    raw = backend.tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True).strip()
    queries = parse_generated_queries(raw, count)
    if len(queries) < count:
        for fallback in synthetic_texts(tool, {"description": description, "category": "tool", "aliases": []}, count):
            if fallback not in queries:
                queries.append(fallback)
            if len(queries) >= count:
                break
    return queries[:count], raw


def generated_query_file_path(args: argparse.Namespace) -> Path:
    return args.generated_query_file or (args.output_dir / "generated_queries.json")


def write_generated_queries(tool_docs: KnowledgePackage, args: argparse.Namespace, progress: ProgressLogger) -> dict[str, Any]:
    path = generated_query_file_path(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        progress.log("generated_queries_loaded", path=str(path), sha256=sha256_file(path), generation_mode=payload.get("generation_mode"))
        return payload
    if args.generation_mode != "llm_from_description":
        raise RuntimeError("generated query file should only be written for llm_from_description")
    backend = _load_local_backend(str(args.generator_model_path), "auto")
    records: dict[str, Any] = {}
    docs = tool_doc_artifacts(tool_docs)
    progress.log("llm_generation_begin", tool_count=len(docs), model_path=str(args.generator_model_path), synthetic_per_tool=args.synthetic_per_tool)
    for idx, doc in enumerate(docs, 1):
        tool, info = description_for(doc)
        queries, raw = local_generate_queries(
            backend,
            tool=tool,
            description=info.get("description", ""),
            count=args.synthetic_per_tool,
            temperature=args.generator_temperature,
            max_new_tokens=args.generator_max_new_tokens,
        )
        records[tool] = {"description": info.get("description", ""), "queries": queries, "raw": raw}
        if idx % 100 == 0:
            save_json(path, {
                "generation_mode": "llm_from_description",
                "generator_model_path": str(args.generator_model_path),
                "generator_model_id": "Llama-3.1-8B-Instruct",
                "temperature": args.generator_temperature,
                "prompt_template": GEN_PROMPT_TEMPLATE,
                "prompt_sha256": prompt_hash(GEN_PROMPT_TEMPLATE),
                "synthetic_per_tool": args.synthetic_per_tool,
                "records": records,
                "complete": False,
            })
            progress.log("llm_generation_progress", generated_tools=idx, total_tools=len(docs))
    payload = {
        "generation_mode": "llm_from_description",
        "generator_model_path": str(args.generator_model_path),
        "generator_model_id": "Llama-3.1-8B-Instruct",
        "temperature": args.generator_temperature,
        "prompt_template": GEN_PROMPT_TEMPLATE,
        "prompt_sha256": prompt_hash(GEN_PROMPT_TEMPLATE),
        "synthetic_per_tool": args.synthetic_per_tool,
        "records": records,
        "complete": True,
    }
    save_json(path, payload)
    progress.log("llm_generation_done", path=str(path), sha256=sha256_file(path), generated_tools=len(records))
    return payload


def build_llm_synthetic_package(tool_docs: KnowledgePackage, payload: dict[str, Any], *, count: int) -> KnowledgePackage:
    artifacts: list[KnowledgeArtifact] = []
    records = payload.get("records") or {}
    for doc in tool_doc_artifacts(tool_docs):
        tool, info = description_for(doc)
        queries = list((records.get(tool) or {}).get("queries") or [])[:count]
        if len(queries) < count:
            queries.extend(synthetic_texts(tool, info, count - len(queries)))
        for idx, text in enumerate(queries[:count]):
            artifacts.append(
                KnowledgeArtifact(
                    signature=f"d3_llm_synthetic::{tool}::{idx}",
                    text=str(text),
                    structured_payload={"type": "synthetic_query", "tool": tool, "tool_description": info["description"], "generation_mode": "llm_from_description"},
                    metadata={"tool": tool, "artifact_origin": "synthetic_doc2query_llm", "synthetic_index": idx},
                )
            )
    return KnowledgePackage(source_id="d3_symmetric_llm_synthetic", artifacts=artifacts, metadata={"generation_mode": "llm_from_description", "synthetic_per_tool": count})


def normalized_text(text: str) -> str:
    return " ".join(re.findall(r"[A-Za-z0-9_]+", text.lower()))


def generated_leak_report(synthetic: KnowledgePackage, queries: list[Any], query_embeddings: list[list[float]], jina_client: JinaAIClient, embed_model: str) -> dict[str, Any]:
    gen_texts = [artifact.text for artifact in synthetic.artifacts]
    gen_norms = {normalized_text(text) for text in gen_texts}
    eval_norms = {normalized_text(item.query) for item in queries}
    exact = len(gen_norms & eval_norms)
    gen_embeddings = batched_query_embeddings(jina_client, gen_texts, embed_model)
    gen_matrix = np.asarray(gen_embeddings, dtype=np.float32)
    eval_matrix = np.asarray(query_embeddings, dtype=np.float32)
    if gen_matrix.size:
        gen_norm = np.linalg.norm(gen_matrix, axis=1)
        gen_norm[gen_norm == 0.0] = 1.0
        gen_matrix = gen_matrix / gen_norm[:, None]
    if eval_matrix.size:
        eval_norm = np.linalg.norm(eval_matrix, axis=1)
        eval_norm[eval_norm == 0.0] = 1.0
        eval_matrix = eval_matrix / eval_norm[:, None]
    near095 = 0
    near090 = 0
    max_sims: list[float] = []
    for start in range(0, len(gen_matrix), 512):
        sims = gen_matrix[start:start + 512] @ eval_matrix.T
        max_batch = sims.max(axis=1) if sims.size else np.asarray([], dtype=np.float32)
        max_sims.extend(float(x) for x in max_batch)
        near095 += int((max_batch >= 0.95).sum())
        near090 += int((max_batch >= 0.90).sum())
    return {"generated_query_count": len(gen_texts), "exact_matches_eval": exact, "near_dup_095_count": near095, "near_dup_090_count": near090, "max_similarity_max": max(max_sims) if max_sims else 0.0}


def build_synthetic_package(tool_docs: KnowledgePackage, *, count: int) -> KnowledgePackage:
    artifacts: list[KnowledgeArtifact] = []
    for doc in tool_doc_artifacts(tool_docs):
        tool, info = description_for(doc)
        for idx, text in enumerate(synthetic_texts(tool, info, count)):
            artifacts.append(
                KnowledgeArtifact(
                    signature=f"d3_synthetic::{tool}::{idx}",
                    text=text,
                    structured_payload={"type": "synthetic_query", "tool": tool, "tool_description": info["description"], "generation_mode": "template_from_description"},
                    metadata={"tool": tool, "artifact_origin": "synthetic_doc2query_template", "synthetic_index": idx},
                )
            )
    return KnowledgePackage(source_id="d3_symmetric_synthetic", artifacts=artifacts, metadata={"generation_mode": "template_from_description", "synthetic_per_tool": count})


def clone_artifact(artifact: KnowledgeArtifact, signature: str | None = None) -> KnowledgeArtifact:
    return KnowledgeArtifact(
        signature=signature or artifact.signature,
        text=artifact.text,
        structured_payload=dict(artifact.structured_payload or {}),
        metadata=dict(artifact.metadata or {}),
    )




def package_with_docs(tool_docs: KnowledgePackage, package: KnowledgePackage, *, source_id: str) -> KnowledgePackage:
    artifacts = [clone_artifact(a, f"d3_tool_doc::{a.signature}") for a in tool_doc_artifacts(tool_docs)]
    artifacts.extend(clone_artifact(a) for a in package.artifacts)
    return KnowledgePackage(source_id=source_id, artifacts=artifacts, metadata={**dict(package.metadata or {}), "includes_tool_descriptions": True})

def merge_uncapped(synthetic: KnowledgePackage, shared: KnowledgePackage) -> KnowledgePackage:
    artifacts = [clone_artifact(a) for a in synthetic.artifacts]
    artifacts.extend(clone_artifact(a, f"d3_experience::{a.signature}") for a in experience_artifacts(shared))
    return KnowledgePackage(source_id="d3_synthetic_plus_experience_uncapped", artifacts=artifacts, metadata={"arm": "synthetic_plus_experience"})


def merge_capped(synthetic: KnowledgePackage, shared: KnowledgePackage, cap: int) -> KnowledgePackage:
    synth_by_tool: dict[str, list[KnowledgeArtifact]] = {}
    exp_by_tool: dict[str, list[KnowledgeArtifact]] = {}
    for artifact in synthetic.artifacts:
        synth_by_tool.setdefault(str(artifact.metadata.get("tool")), []).append(artifact)
    for artifact in experience_artifacts(shared):
        exp_by_tool.setdefault(str(artifact.metadata.get("tool")), []).append(artifact)
    artifacts: list[KnowledgeArtifact] = []
    for tool in sorted(synth_by_tool):
        selected = [clone_artifact(a, f"d3_capped_exp::{a.signature}") for a in exp_by_tool.get(tool, [])[:cap]]
        remaining = max(0, cap - len(selected))
        selected.extend(clone_artifact(a, f"d3_capped_synth::{a.signature}") for a in synth_by_tool[tool][:remaining])
        artifacts.extend(selected[:cap])
    return KnowledgePackage(source_id="d3_synthetic_plus_experience_capped", artifacts=artifacts, metadata={"arm": "capped_synthetic_plus_experience", "cap_per_tool": cap})


def prepare_index(package: KnowledgePackage, retriever: str, jina_client: JinaAIClient, embed_model: str, bge_model: Any | None):
    candidates = package_to_candidates(package, jina_client, embed_model)
    texts = [candidate.text for candidate in candidates]
    if retriever == "jina":
        index = normalize_matrix([candidate.embedding for candidate in candidates]) if candidates else np.zeros((0, 0), dtype=np.float32)
    elif retriever == "bm25":
        index = BM25Index(texts)
    elif retriever == "bge":
        if bge_model is None:
            raise RuntimeError("BGE retriever requested without BGE model")
        index = bge_encode(bge_model, texts)
    else:
        raise ValueError(f"unknown retriever: {retriever}")
    return candidates, index


def score_query(index: Any, retriever: str, query: str, jina_embedding: list[float] | None, bge_model: Any | None) -> np.ndarray:
    if retriever == "bm25":
        return index.encode_query(query)
    if retriever == "bge":
        q = bge_encode(bge_model, [query])[0]
        return index @ q if getattr(index, "size", 0) else np.asarray([], dtype=np.float32)
    q = np.asarray(jina_embedding, dtype=np.float32)
    norm = float(np.linalg.norm(q))
    if norm > 0:
        q = q / norm
    return index @ q if getattr(index, "size", 0) else np.asarray([], dtype=np.float32)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "recall_at_5": sum(1 for row in rows if row["gold_in_top_k"]) / len(rows) if rows else 0.0,
        "retrieval_top1": sum(1 for row in rows if (row["candidate_tools"] or [""])[0] in set(row["gold_tools"])) / len(rows) if rows else 0.0,
        "mean_distinct_tools": statistics.mean(row["distinct_tool_count"] for row in rows) if rows else 0.0,
        "lt5_share": sum(1 for row in rows if row["distinct_tool_count"] < 5) / len(rows) if rows else 0.0,
    }


def evaluate(package: KnowledgePackage, *, arm: str, retriever: str, test_items: list[Any], heldout_tools: set[str], jina_embeddings: list[list[float] | None], jina_client: JinaAIClient, embed_model: str, bge_model: Any | None, args: argparse.Namespace, progress: ProgressLogger, seed: int) -> dict[str, Any]:
    progress.log("arm_prepare_begin", seed=seed, arm=arm, retriever=retriever, artifact_count=len(package.artifacts))
    started = time.perf_counter()
    candidates, index = prepare_index(package, retriever, jina_client, embed_model, bge_model)
    progress.log("arm_prepare_done", seed=seed, arm=arm, retriever=retriever, elapsed_s=time.perf_counter() - started)
    rows = []
    for item, jina_embedding in zip(test_items, jina_embeddings):
        scores = score_query(index, retriever, item.query, jina_embedding, bge_model)
        ranked, pool_tools = build_ranked_candidates(candidates, scores, args.retrieval_pool_size, args.top_k, args.retrieval_mode)
        candidate_tools = list(dict.fromkeys(candidate.parent_tool for candidate in ranked))[: args.top_k]
        if len(candidate_tools) < args.top_k:
            raise RuntimeError(f"{arm}/{retriever}/seed{seed} query {item.query_id} returned {len(candidate_tools)} distinct tools")
        gold = item.gold_tools
        rows.append(
            {
                "query_id": item.query_id,
                "query_text": item.query,
                "group": item.group,
                "gold_tools": gold,
                "candidate_tools": candidate_tools,
                "candidate_ids": [candidate.candidate_id for candidate in ranked],
                "gold_in_top_k": any(tool in gold for tool in candidate_tools),
                "subset": "heldout" if gold and all(tool in heldout_tools for tool in gold) else "labeled",
                "distinct_tool_count": len(candidate_tools),
                "retrieval_pool_tools": pool_tools,
            }
        )
    result = {
        "paper_eligible": True,
        "seed": seed,
        "arm": arm,
        "retriever": retriever,
        "candidate_rule": "distinct5_walkdown",
        "retrieval_pool_size": args.retrieval_pool_size,
        "top_k": args.top_k,
        "package_sha256": package_hash(package),
        "index_document_count": len(package.artifacts),
        "rows": rows,
        "metrics": summarize(rows),
        "subset_metrics": {
            "heldout": summarize([row for row in rows if row["subset"] == "heldout"]),
            "labeled": summarize([row for row in rows if row["subset"] == "labeled"]),
        },
    }
    progress.log("arm_done", seed=seed, arm=arm, retriever=retriever, heldout_recall_at_5=result["subset_metrics"]["heldout"]["recall_at_5"], labeled_recall_at_5=result["subset_metrics"]["labeled"]["recall_at_5"])
    return result


def aggregate(results: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in sorted({"/".join(k.split("/")[:2]) for k in results}):
        vals = [v for k, v in results.items() if k.startswith(key + "/")]
        out[key] = {
            subset: {
                metric: {
                    "mean": statistics.mean(v["subset_metrics"][subset][metric] for v in vals),
                    "sd": statistics.stdev(v["subset_metrics"][subset][metric] for v in vals) if len(vals) > 1 else 0.0,
                    "seeds": [v["seed"] for v in vals],
                }
                for metric in ("recall_at_5", "retrieval_top1", "mean_distinct_tools", "lt5_share")
            }
            for subset in ("heldout", "labeled")
        }
    return out


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    commit, dirty = assert_clean_tree(allow_dirty=args.allow_dirty)
    progress = ProgressLogger(args.output_dir)
    progress.log("launch", repo_commit=commit, dirty_entry_count=len(dirty), generation_mode=args.generation_mode)
    seeds = [int(x) for x in parse_csv(args.seeds)]
    seed_dirs = parse_seed_dirs(args.e6_dirs)
    retrievers = parse_csv(args.retrievers)
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
        query_embeddings = batched_query_embeddings(jina_client, [item.query for item in queries], args.embed_model)
        progress.log("data_ready", test_count=len(queries), heldout_tool_count=len(heldout_tools), heldout_tools_sha256=heldout_sha, junk_filter=junk_info, eval_filter=eval_filter)

        bge_model = None
        if "bge" in retrievers:
            progress.log("load_bge_begin", model_path=str(args.bge_model_path))
            device = os.environ.get("BGE_DEVICE") or ("cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
            bge_model = build_bge_encoder(args.bge_model_path, device=device)
            progress.log("load_bge_done", model_sha256=sha256_file(args.bge_model_path / "config.json"), device=device)

        generation_payload: dict[str, Any] | None = None
        all_results: dict[str, Any] = {}
        for seed in seeds:
            seed_dir = seed_dirs[seed]
            package_dir = seed_dir / f"seed_{seed}" / "packages" if (seed_dir / f"seed_{seed}" / "packages").exists() else seed_dir / "packages"
            tool_docs, tool_docs_sha = load_package_file(package_dir / "tool_docs.json")
            synapse, synapse_sha = load_package_file(package_dir / "synapse_shared.json")
            if args.generation_mode == "llm_from_description":
                if generation_payload is None:
                    generation_payload = write_generated_queries(tool_docs, args, progress)
                synthetic = build_llm_synthetic_package(tool_docs, generation_payload, count=max(args.synthetic_per_tool, args.cap_per_tool))
            else:
                synthetic = build_synthetic_package(tool_docs, count=max(args.synthetic_per_tool, args.cap_per_tool))
            leak_report = generated_leak_report(synthetic, queries, query_embeddings, jina_client, args.embed_model)
            if args.generation_mode == "llm_from_description" and (leak_report["exact_matches_eval"] != 0 or leak_report["near_dup_095_count"] != 0):
                raise RuntimeError(f"generated queries violate exposure contract: {leak_report}")
            packages = {
                "synthetic_docs": package_with_docs(tool_docs, synthetic, source_id="d3_synthetic_docs_with_descriptions"),
                "synthetic_plus_experience": package_with_docs(tool_docs, merge_uncapped(synthetic, synapse), source_id="d3_synthetic_plus_experience_with_descriptions"),
                "capped_synthetic_plus_experience": package_with_docs(tool_docs, merge_capped(synthetic, synapse, args.cap_per_tool), source_id="d3_capped_synthetic_plus_experience_with_descriptions"),
            }
            progress.log("seed_packages_ready", seed=seed, source_dir=str(package_dir), tool_docs_sha256=tool_docs_sha, synapse_sha256=synapse_sha, synthetic_artifacts=len(synthetic.artifacts), capped_artifacts=len(packages["capped_synthetic_plus_experience"].artifacts), generated_leak_report=leak_report)
            for retriever in retrievers:
                for arm, package in packages.items():
                    result = evaluate(package, arm=arm, retriever=retriever, test_items=queries, heldout_tools=heldout_set, jina_embeddings=query_embeddings, jina_client=jina_client, embed_model=args.embed_model, bge_model=bge_model, args=args, progress=progress, seed=seed)
                    result.update(
                        {
                            "repo_commit": commit,
                            "data_mode": "toolbench_train",
                            "heldout_tools_sha256": heldout_sha,
                            "heldout_sanity": heldout_sanity,
                            "generation": {
                                "mode": args.generation_mode,
                                "synthetic_per_tool": args.synthetic_per_tool,
                                "cap_per_tool": args.cap_per_tool,
                                "generated_query_file": str(generated_query_file_path(args)) if args.generation_mode == "llm_from_description" else None,
                                "generated_query_file_sha256": sha256_file(generated_query_file_path(args)) if args.generation_mode == "llm_from_description" and generated_query_file_path(args).exists() else None,
                                "generator_model_id": "Llama-3.1-8B-Instruct" if args.generation_mode == "llm_from_description" else None,
                                "prompt_sha256": prompt_hash(GEN_PROMPT_TEMPLATE) if args.generation_mode == "llm_from_description" else None,
                                "leak_report": leak_report,
                                "note": "Synthetic query documents are generated from tool descriptions only; no client experience, eval queries, or held-out list are used.",
                            },
                            "source": {
                                "e6_seed_dir": str(seed_dir),
                                "package_dir": str(package_dir),
                                "tool_docs_sha256": tool_docs_sha,
                                "synapse_sha256": synapse_sha,
                            },
                        }
                    )
                    out_dir = args.output_dir / f"seed_{seed}" / retriever
                    save_json(out_dir / f"{arm}.json", result)
                    all_results[f"{arm}/{retriever}/{seed}"] = result
        config_payload = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        summary = {
            "paper_eligible": True,
            "repo_commit": commit,
            "config": config_payload | {"heldout_tools_sha256": heldout_sha, "local_embedder": embedder_info},
            "aggregate": aggregate(all_results),
        }
        save_json(args.output_dir / "summary.json", summary)
        progress.log("complete", output_dir=str(args.output_dir))
        print(json.dumps(summary["aggregate"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
