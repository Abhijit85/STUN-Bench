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
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from math_qa import JinaAIClient
from scripts.model_paths import default_llama31_8b_path
from scripts.run_gsm8k_small_router_sweep import (  # noqa: E402
    LocalChatBackend,
    _load_local_backend,
    build_credentials,
    cosine_similarity,
    gold_route_label,
    labels_match,
    load_records,
    normalize_label,
    parse_seed_list,
    query_text,
    sample_records,
)
from scripts.run_toolbench_retrieval_at_scale import (  # noqa: E402
    DEFAULT_QUERY_FILE as TOOLBENCH_QUERY_FILE,
    DEFAULT_TOOL_DOC_DIR,
    QueryRecord,
    ToolDoc,
    ToolScenario,
    flatten_scenarios,
    load_tool_docs_from_paths,
    merge_tool_docs,
    parse_toolllama_eval_queries,
)
from synapse.runtime import SynapseRuntime

DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "verification" / "reranker_prompt_sweep"
DEFAULT_MODEL_PATH = default_llama31_8b_path()


@dataclass
class RoutedCandidate:
    candidate_id: str
    label: str
    parent_tool: str
    when_to_use: list[str]
    do_not_use_when: list[str]
    text: str
    embedding: list[float]
    provenance: str


@dataclass
class PromptResult:
    predicted_tool: str
    predicted_candidate: str
    parse_ok: bool
    fallback_used: bool
    latency_seconds: float
    margin: float | None
    raw_output: str
    prompt_hash: str


GSM8K_DEMOS = [
    {
        "query": "A factory makes 120 bolts per hour for 3 hours. How many bolts does it make?",
        "tool": "General Logic and Counting",
        "why": "Handles simple arithmetic and everyday multi-step counting problems.",
        "avoid": "Do not use when the core task is profit, interest, discounts, or transaction value.",
    },
    {
        "query": "A shop buys bracelets for $4 each and sells them for $7 each. What profit does it make on 20 bracelets?",
        "tool": "Financial and Banking Calculator",
        "why": "Handles gains, losses, prices, revenue, and profit calculations.",
        "avoid": "Do not use when the query is purely geometric or asks only for distance or area.",
    },
    {
        "query": "Two workers can paint a fence in 6 hours together. If one worker alone takes 10 hours, how long would the other take?",
        "tool": "Work, Rate, and Time Analyzer",
        "why": "Handles rate, time, speed, and combined-work problems.",
        "avoid": "Do not use when the question is only about percentages or financial value comparison.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the V0-V3 reranker prompt sweep on GSM8K and ToolBench.")
    parser.add_argument("--benchmarks", type=str, default="gsm8k,toolbench")
    parser.add_argument("--variants", type=str, default="V0,V1,V2,V3")
    parser.add_argument("--gsm8k-sample-count", type=int, default=100)
    parser.add_argument("--gsm8k-seeds", type=str, default="42,123,456,789,1024")
    parser.add_argument("--toolbench-query-count", type=int, default=200)
    parser.add_argument("--toolbench-query-seed", type=int, default=42)
    parser.add_argument("--toolbench-subset-seeds", type=str, default="1")
    parser.add_argument("--toolbench-query-file", type=Path, default=TOOLBENCH_QUERY_FILE)
    parser.add_argument("--toolbench-tool-doc-dir", type=Path, default=DEFAULT_TOOL_DOC_DIR)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--client-count", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--embed-model", type=str, default="jina-embeddings-v2-base-en")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--max-tokens", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def short_text(value: str, limit: int = 180) -> str:
    text = re.sub(r"\s+", " ", (value or "").strip())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def render_precautions(precautions: list[str], conflict_log: list[str]) -> list[str]:
    merged: list[str] = []
    for item in precautions + conflict_log:
        cleaned = short_text(str(item), limit=160)
        if cleaned and cleaned not in merged:
            merged.append(cleaned)
    return merged


def hash_prompt(payload: dict[str, Any]) -> str:
    material = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def load_local_backend(model_path: str) -> LocalChatBackend:
    return _load_local_backend(model_path, "auto")


def local_generate(backend: LocalChatBackend, prompt: str, max_tokens: int) -> str:
    messages = [{"role": "user", "content": prompt}]
    rendered = backend.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = backend.tokenizer(rendered, return_tensors="pt")
    encoded = {key: value.to(backend.model.device) for key, value in encoded.items()}
    with torch_no_grad():
        output_ids = backend.model.generate(
            **encoded,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=backend.tokenizer.pad_token_id,
            eos_token_id=backend.tokenizer.eos_token_id,
        )
    prompt_len = encoded["input_ids"].shape[1]
    return backend.tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True).strip()


def torch_no_grad():
    import torch

    return torch.no_grad()


def next_token_scores(backend: LocalChatBackend, prompt: str, options: list[str]) -> tuple[str, dict[str, float], float]:
    import torch

    messages = [{"role": "user", "content": prompt}]
    rendered = backend.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = backend.tokenizer(rendered, return_tensors="pt")
    encoded = {key: value.to(backend.model.device) for key, value in encoded.items()}
    with torch.no_grad():
        logits = backend.model(**encoded).logits[0, -1]
    scores: dict[str, float] = {}
    for option in options:
        token_ids = backend.tokenizer.encode(option, add_special_tokens=False)
        if not token_ids:
            continue
        scores[option] = float(logits[token_ids[-1]].item())
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if not ordered:
        return "", {}, 0.0
    top = ordered[0][0]
    margin = ordered[0][1] - ordered[1][1] if len(ordered) > 1 else math.inf
    return top, scores, margin


def build_gsm8k_candidates(rounds: int, client_count: int, jina_client: JinaAIClient, embed_model: str) -> list[RoutedCandidate]:
    runtime = SynapseRuntime.build_local_runtime(REPO_ROOT, build_credentials(), client_count=client_count)
    for _ in range(max(1, rounds)):
        runtime.run_round()

    candidates: list[RoutedCandidate] = []
    texts: list[str] = []
    for artifact in runtime.server.compendium.build_snapshot().artifacts:
        metadata = artifact.metadata or {}
        if metadata.get("tool") != "mathqa":
            continue
        label = str(metadata.get("scenario") or metadata.get("domain") or "").strip()
        if not label:
            continue
        payload = artifact.structured_payload or {}
        scenario_context = str(payload.get("scenario_context") or "").strip()
        annex_summary = str(payload.get("annex_summary") or "").strip()
        precautions = [str(item).strip() for item in payload.get("precautions", []) if str(item).strip()] if isinstance(payload.get("precautions"), list) else []
        conflict_log = [str(item).strip() for item in payload.get("conflict_log", []) if str(item).strip()] if isinstance(payload.get("conflict_log"), list) else []
        when_to_use = [item for item in [scenario_context, annex_summary] if item]
        do_not_use = render_precautions(precautions, conflict_log)
        text = str(artifact.text or "").strip()
        candidates.append(
            RoutedCandidate(
                candidate_id=label,
                label=label,
                parent_tool=label,
                when_to_use=when_to_use,
                do_not_use_when=do_not_use,
                text=text,
                embedding=[],
                provenance="gsm8k_runtime_snapshot",
            )
        )
        texts.append(text)
    prev = os.environ.get("JINA_EMBED_MODEL")
    os.environ["JINA_EMBED_MODEL"] = embed_model
    try:
        vectors = jina_client.get_embeddings(texts)
    finally:
        if prev is None:
            os.environ.pop("JINA_EMBED_MODEL", None)
        else:
            os.environ["JINA_EMBED_MODEL"] = prev
    for candidate, vector in zip(candidates, vectors):
        candidate.embedding = vector
    return candidates


def build_toolbench_dataset(query_file: Path, tool_doc_dir: Path, query_count: int, query_seed: int, jina_client: JinaAIClient, embed_model: str) -> tuple[list[QueryRecord], list[RoutedCandidate], list[dict[str, str]]]:
    queries, extracted_tool_docs, _meta = parse_toolllama_eval_queries(query_file, query_count=query_count, query_seed=query_seed, query_ids=None)
    local_tool_docs = load_tool_docs_from_paths(tool_doc_dir, None)
    tool_docs: dict[str, ToolDoc] = merge_tool_docs(primary=local_tool_docs, fallback=extracted_tool_docs)
    scenarios = flatten_scenarios(list(tool_docs.keys()), tool_docs)
    candidates: list[RoutedCandidate] = []
    texts: list[str] = []
    for scenario in scenarios:
        doc = tool_docs[scenario.parent_tool]
        when_to_use = [scenario.text]
        candidates.append(
            RoutedCandidate(
                candidate_id=scenario.scenario_id,
                label=scenario.scenario_name,
                parent_tool=scenario.parent_tool,
                when_to_use=when_to_use,
                do_not_use_when=[],
                text=scenario.text,
                embedding=[],
                provenance=scenario.provenance,
            )
        )
        texts.append(scenario.text)
    prev = os.environ.get("JINA_EMBED_MODEL")
    os.environ["JINA_EMBED_MODEL"] = embed_model
    try:
        vectors = jina_client.get_embeddings(texts)
    finally:
        if prev is None:
            os.environ.pop("JINA_EMBED_MODEL", None)
        else:
            os.environ["JINA_EMBED_MODEL"] = prev
    for candidate, vector in zip(candidates, vectors):
        candidate.embedding = vector

    eval_ids = {query.query_id for query in queries}
    demo_query_count = min(300, query_count + 20)
    try:
        all_queries, _, _ = parse_toolllama_eval_queries(
            query_file,
            query_count=demo_query_count,
            query_seed=0,
            query_ids=None,
        )
    except ValueError:
        # The local checkout only exposes 293 parsable eval queries. When the
        # evaluation slice already uses the full set, reuse it for demo mining.
        all_queries = queries
    demo_records: list[dict[str, str]] = []
    for query in all_queries:
        if query.query_id in eval_ids:
            continue
        gold = query.gold_parent_tools[0] if query.gold_parent_tools else ""
        if not gold:
            continue
        demo_records.append({
            "query": query.query_text,
            "tool": gold,
            "why": f"Use when the request matches {gold.replace('_', ' ')}.",
            "avoid": "Do not use when another tool family is a tighter semantic match.",
        })
        if len(demo_records) == 3:
            break
    return queries, candidates, demo_records


def build_query_embeddings(jina_client: JinaAIClient, texts: list[str], embed_model: str) -> list[list[float]]:
    prev = os.environ.get("JINA_EMBED_MODEL")
    os.environ["JINA_EMBED_MODEL"] = embed_model
    try:
        return jina_client.get_embeddings(texts)
    finally:
        if prev is None:
            os.environ.pop("JINA_EMBED_MODEL", None)
        else:
            os.environ["JINA_EMBED_MODEL"] = prev


def grouped_tools(candidates: list[RoutedCandidate]) -> list[tuple[str, list[RoutedCandidate]]]:
    grouped: dict[str, list[RoutedCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.parent_tool, []).append(candidate)
    return list(grouped.items())


def parse_tool_summary(candidate: RoutedCandidate) -> tuple[str, str, list[str]]:
    description = ""
    endpoints = ""
    scenarios: list[str] = []
    if candidate.when_to_use:
        summary = candidate.when_to_use[0]
        parts = [part.strip() for part in summary.split(";") if part.strip()]
        for part in parts:
            if part.startswith("apis="):
                endpoints = part.split("=", 1)[1].strip()
            elif part.startswith("docs="):
                description = part.split("=", 1)[1].strip()
        for extra in candidate.when_to_use[1:]:
            cleaned = short_text(extra, 180)
            if cleaned and cleaned not in scenarios:
                scenarios.append(cleaned)
    if not description:
        description = short_text(candidate.text, 180)
    return description, endpoints, scenarios


def render_variant_prompt(variant: str, benchmark: str, query: str, candidates: list[RoutedCandidate], demos: list[dict[str, str]]) -> tuple[str, dict[str, Any], list[tuple[str, list[RoutedCandidate]]]]:
    domain_open = "You route a user query to the single most appropriate tool."
    if variant == "V0":
        options = [f"{index}. {candidate.label}\nContext: {candidate.text}" for index, candidate in enumerate(candidates, start=1)]
        prompt = (
            "You are a routing reranker for user queries.\n"
            "Choose the single best scenario label for the query from the candidate list.\n"
            "Return only the exact scenario label, with no explanation.\n\n"
            f"Query:\n{query}\n\n"
            f"Candidates:\n{chr(10).join(options)}\n"
        )
        return prompt, {"variant": variant, "benchmark": benchmark, "mode": "scenario_label_generation"}, []

    grouped = grouped_tools(candidates) if variant == "V3" else []
    demo_block = ""
    if variant == "V2":
        demo_lines = []
        for demo_index, demo in enumerate(demos[:3], start=1):
            demo_lines.extend(
                [
                    f"Demo {demo_index}:",
                    f"Query: {demo['query']}",
                    f"Answer: 1  (tool: {demo['tool']}; when to use: {demo['why']}; do not use when: {demo['avoid']})",
                    "",
                ]
            )
        demo_block = "\n".join(demo_lines)

    if variant in {"V1", "V2"}:
        options = []
        for index, candidate in enumerate(candidates, start=1):
            description, endpoints, scenarios = parse_tool_summary(candidate)
            scenario_text = "; ".join(scenarios[:2]) or short_text(candidate.text, 180)
            avoid = "; ".join(candidate.do_not_use_when[:3]) or "—"
            options.append(
                f"[{index}] tool: {candidate.parent_tool}\n"
                f"    description: {description}\n"
                f"    endpoints: {endpoints or '—'}\n"
                f"    scenarios: {scenario_text}\n"
                f"    do not use when: {avoid}"
            )
        prompt = (
            f"{domain_open}\n"
            "Each candidate below is a tool suggested by the shared compendium.\n"
            "Read the tool description and endpoints first, then use the scenarios and precautions as supporting evidence.\n"
            "Pick the candidate whose TOOL is the right one for the query.\n"
            "Answer with the candidate number only.\n\n"
            f"{demo_block}"
            f"Query:\n{query}\n\n"
            f"Candidates:\n{chr(10).join(options)}\n\n"
            "Answer:\n"
        )
        return prompt, {"variant": variant, "benchmark": benchmark, "mode": "candidate_index_generation"}, []

    if variant == "V3":
        lines = []
        for index, (tool_name, tool_candidates) in enumerate(grouped, start=1):
            description = ""
            endpoints = ""
            scenarios: list[str] = []
            precaution_union: list[str] = []
            for candidate in tool_candidates[:2]:
                candidate_description, candidate_endpoints, candidate_scenarios = parse_tool_summary(candidate)
                if not description and candidate_description:
                    description = candidate_description
                if not endpoints and candidate_endpoints:
                    endpoints = candidate_endpoints
                for scenario in candidate_scenarios[:2]:
                    if scenario not in scenarios:
                        scenarios.append(scenario)
                for item in candidate.do_not_use_when:
                    if item not in precaution_union:
                        precaution_union.append(item)
            scenario_text = "; ".join(short_text(item, 140) for item in scenarios[:2]) or short_text(tool_candidates[0].text, 140)
            avoid = "; ".join(precaution_union[:3]) or "—"
            lines.append(
                f"[{index}] {tool_name}\n"
                f"    description: {description or short_text(tool_candidates[0].text, 140)}\n"
                f"    endpoints: {endpoints or '—'}\n"
                f"    scenarios: {scenario_text}\n"
                f"    do not use when: {avoid}"
            )
        prompt = (
            f"{domain_open}\n"
            "Below are candidate tools suggested by the compendium.\n"
            "For each tool, read the description and endpoints first, then use the scenarios and precautions as supporting evidence.\n"
            "Answer with the tool number only.\n\n"
            f"Query:\n{query}\n\n"
            f"Tools:\n{chr(10).join(lines)}\n\n"
            "Answer:\n"
        )
        return prompt, {"variant": variant, "benchmark": benchmark, "mode": "tool_index_logit"}, grouped

    raise ValueError(f"Unknown variant: {variant}")


def run_prompt(backend: LocalChatBackend, variant: str, benchmark: str, query: str, candidates: list[RoutedCandidate], demos: list[dict[str, str]], top_candidate: RoutedCandidate) -> PromptResult:
    prompt, prompt_meta, grouped = render_variant_prompt(variant, benchmark, query, candidates, demos)
    prompt_hash = hash_prompt(prompt_meta | {"prompt": prompt})
    started = time.perf_counter()
    if variant == "V3":
        option_labels = [str(index) for index in range(1, len(grouped) + 1)]
        chosen, _scores, margin = next_token_scores(backend, prompt, option_labels)
        latency = time.perf_counter() - started
        if not chosen:
            return PromptResult(top_candidate.parent_tool, top_candidate.candidate_id, False, True, latency, None, "", prompt_hash)
        idx = int(chosen) - 1
        tool_name, tool_candidates = grouped[idx]
        return PromptResult(tool_name, tool_candidates[0].candidate_id, True, False, latency, margin, chosen, prompt_hash)

    raw_output = local_generate(backend, prompt, max_tokens=12)
    latency = time.perf_counter() - started
    first_line = raw_output.strip().splitlines()[0].strip() if raw_output.strip() else ""
    if variant == "V0":
        for candidate in candidates:
            if labels_match(first_line, candidate.label):
                return PromptResult(candidate.parent_tool, candidate.candidate_id, True, False, latency, None, raw_output, prompt_hash)
            if normalize_label(candidate.label) in normalize_label(first_line):
                return PromptResult(candidate.parent_tool, candidate.candidate_id, True, False, latency, None, raw_output, prompt_hash)
        return PromptResult(top_candidate.parent_tool, top_candidate.candidate_id, False, True, latency, None, raw_output, prompt_hash)

    match = re.search(r"\b([1-9])\b", first_line)
    if not match:
        return PromptResult(top_candidate.parent_tool, top_candidate.candidate_id, False, True, latency, None, raw_output, prompt_hash)
    idx = int(match.group(1)) - 1
    if idx < 0 or idx >= len(candidates):
        return PromptResult(top_candidate.parent_tool, top_candidate.candidate_id, False, True, latency, None, raw_output, prompt_hash)
    chosen = candidates[idx]
    return PromptResult(chosen.parent_tool, chosen.candidate_id, True, False, latency, None, raw_output, prompt_hash)


def summarize(seed_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    accuracies = [payload["accuracy"] for payload in seed_payloads]
    latencies = [payload["mean_latency_seconds"] for payload in seed_payloads]
    parse_fail_rates = [payload["parse_failure_rate"] for payload in seed_payloads]
    fallback_rates = [payload["fallback_rate"] for payload in seed_payloads]
    return {
        "mean_accuracy": statistics.mean(accuracies) if accuracies else 0.0,
        "sd_accuracy": statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
        "mean_latency_seconds": statistics.mean(latencies) if latencies else 0.0,
        "sd_latency_seconds": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        "mean_parse_failure_rate": statistics.mean(parse_fail_rates) if parse_fail_rates else 0.0,
        "mean_fallback_rate": statistics.mean(fallback_rates) if fallback_rates else 0.0,
    }


def evaluate_gsm8k_variant(variant: str, seeds: list[int], sample_count: int, candidates: list[RoutedCandidate], jina_client: JinaAIClient, embed_model: str, backend: LocalChatBackend) -> dict[str, Any]:
    records = load_records(REPO_ROOT / "GSM8K_500_rebuttal_run" / "GSM8K_500_samples.json")
    query_map = {seed: sample_records(records, seed=seed, sample_count=sample_count) for seed in seeds}
    all_queries = [query_text(record) for seed in seeds for record in query_map[seed]]
    query_embeddings = build_query_embeddings(jina_client, all_queries, embed_model)
    pointer = 0
    seed_payloads: list[dict[str, Any]] = []
    for seed in seeds:
        rows = []
        correct = 0
        parse_failures = 0
        fallback_used = 0
        total_latency = 0.0
        for record in query_map[seed]:
            query = query_text(record)
            gold = gold_route_label(record)
            embedding = query_embeddings[pointer]
            pointer += 1
            ranked = sorted(candidates, key=lambda candidate: cosine_similarity(embedding, candidate.embedding), reverse=True)[:5]
            top_candidate = ranked[0]
            result = run_prompt(backend, variant, "gsm8k", query, ranked, GSM8K_DEMOS, top_candidate)
            hit = labels_match(result.predicted_tool, gold)
            correct += int(hit)
            parse_failures += int(not result.parse_ok)
            fallback_used += int(result.fallback_used)
            total_latency += result.latency_seconds
            rows.append({
                "query_id": record.get("query_id") or record.get("sample_id"),
                "query_text": query,
                "ground_truth_domain": gold,
                "predicted_domain": result.predicted_tool,
                "predicted_candidate": result.predicted_candidate,
                "routed_correctly": hit,
                "parse_ok": result.parse_ok,
                "fallback_used": result.fallback_used,
                "latency_seconds": result.latency_seconds,
                "prompt_hash": result.prompt_hash,
                "margin": result.margin,
                "raw_output": result.raw_output,
                "top_candidates": [candidate.parent_tool for candidate in ranked],
                "top_candidate_ids": [candidate.candidate_id for candidate in ranked],
            })
        seed_payloads.append({
            "seed": seed,
            "sample_count": sample_count,
            "correct": correct,
            "accuracy": correct / sample_count,
            "mean_latency_seconds": total_latency / sample_count,
            "parse_failure_rate": parse_failures / sample_count,
            "fallback_rate": fallback_used / sample_count,
            "rows": rows,
        })
    return {"seed_results": seed_payloads, **summarize(seed_payloads)}


def evaluate_toolbench_variant(variant: str, queries: list[QueryRecord], candidates: list[RoutedCandidate], demos: list[dict[str, str]], jina_client: JinaAIClient, embed_model: str, backend: LocalChatBackend) -> dict[str, Any]:
    query_embeddings = build_query_embeddings(jina_client, [query.query_text for query in queries], embed_model)
    rows = []
    correct = 0
    parse_failures = 0
    fallback_used = 0
    total_latency = 0.0
    for query, embedding in zip(queries, query_embeddings):
        ranked = sorted(candidates, key=lambda candidate: cosine_similarity(embedding, candidate.embedding), reverse=True)[:5]
        top_candidate = ranked[0]
        result = run_prompt(backend, variant, "toolbench", query.query_text, ranked, demos, top_candidate)
        hit = result.predicted_tool in query.gold_parent_tools
        correct += int(hit)
        parse_failures += int(not result.parse_ok)
        fallback_used += int(result.fallback_used)
        total_latency += result.latency_seconds
        rows.append({
            "query_id": query.query_id,
            "query_text": query.query_text,
            "gold_parent_tools": query.gold_parent_tools,
            "predicted_tool": result.predicted_tool,
            "predicted_candidate": result.predicted_candidate,
            "routed_correctly": hit,
            "parse_ok": result.parse_ok,
            "fallback_used": result.fallback_used,
            "latency_seconds": result.latency_seconds,
            "prompt_hash": result.prompt_hash,
            "margin": result.margin,
            "raw_output": result.raw_output,
            "top_candidates": [candidate.parent_tool for candidate in ranked],
            "top_candidate_ids": [candidate.candidate_id for candidate in ranked],
        })
    sample_count = len(queries)
    return {
        "sample_count": sample_count,
        "correct": correct,
        "accuracy": correct / sample_count if sample_count else 0.0,
        "mean_latency_seconds": total_latency / sample_count if sample_count else 0.0,
        "parse_failure_rate": parse_failures / sample_count if sample_count else 0.0,
        "fallback_rate": fallback_used / sample_count if sample_count else 0.0,
        "rows": rows,
    }


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    benchmarks = parse_csv(args.benchmarks)
    variants = parse_csv(args.variants)
    seeds = parse_seed_list(args.gsm8k_seeds)

    backend = load_local_backend(args.model_path)
    jina_client = JinaAIClient(api_keys=os.environ.get("JINA_API_KEY") and [os.environ["JINA_API_KEY"]] or [])

    combined: dict[str, Any] = {
        "benchmarks": benchmarks,
        "variants": variants,
        "model_path": args.model_path,
        "embed_model": args.embed_model,
        "top_k": args.top_k,
    }

    if "gsm8k" in benchmarks:
        gsm8k_candidates = build_gsm8k_candidates(args.rounds, args.client_count, jina_client, args.embed_model)
        benchmark_dir = args.output_dir / "gsm8k"
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        benchmark_summary = {}
        for variant in variants:
            result = evaluate_gsm8k_variant(variant, seeds, args.gsm8k_sample_count, gsm8k_candidates, jina_client, args.embed_model, backend)
            variant_dir = benchmark_dir / variant
            variant_dir.mkdir(parents=True, exist_ok=True)
            for seed_payload in result["seed_results"]:
                (variant_dir / f"routing_seed_{seed_payload['seed']}.json").write_text(json.dumps(seed_payload, indent=2), encoding="utf-8")
            summary = {key: value for key, value in result.items() if key != "seed_results"}
            summary.update({"variant": variant, "benchmark": "gsm8k", "seeds": seeds})
            (variant_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            benchmark_summary[variant] = summary
            print(f"[gsm8k] {variant}: acc={summary['mean_accuracy']:.3f} sd={summary['sd_accuracy']:.3f} parse_fail={summary['mean_parse_failure_rate']:.3f}", flush=True)
        combined["gsm8k"] = benchmark_summary

    if "toolbench" in benchmarks:
        queries, toolbench_candidates, demo_records = build_toolbench_dataset(args.toolbench_query_file, args.toolbench_tool_doc_dir, args.toolbench_query_count, args.toolbench_query_seed, jina_client, args.embed_model)
        benchmark_dir = args.output_dir / "toolbench"
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        benchmark_summary = {}
        for variant in variants:
            result = evaluate_toolbench_variant(variant, queries, toolbench_candidates, demo_records, jina_client, args.embed_model, backend)
            variant_dir = benchmark_dir / variant
            variant_dir.mkdir(parents=True, exist_ok=True)
            (variant_dir / "routing_slice.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            summary = {key: value for key, value in result.items() if key != "rows"}
            summary.update({"variant": variant, "benchmark": "toolbench", "query_count": len(queries)})
            (variant_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            benchmark_summary[variant] = summary
            print(f"[toolbench] {variant}: acc={summary['accuracy']:.3f} parse_fail={summary['parse_failure_rate']:.3f}", flush=True)
        combined["toolbench"] = benchmark_summary

    (args.output_dir / "combined_summary.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "variants": variants, "benchmarks": benchmarks}, indent=2))


if __name__ == "__main__":
    main()
