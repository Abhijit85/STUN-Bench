#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import random
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from populate_vector_store import JinaAIClient


DEFAULT_QUERY_FILE = REPO_ROOT / "external_datasets" / "toolbench" / "toolllama_G123_dfs_eval.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "verification" / "toolbench_retrieval_at_scale"
DEFAULT_TOOL_DOC_DIR = REPO_ROOT / "external_datasets" / "toolbench" / "ToolBench-master" / "data_example" / "toolenv" / "tools"


@dataclass
class ToolScenario:
    parent_tool: str
    scenario_id: str
    scenario_name: str
    text: str
    provenance: str


@dataclass
class ToolDoc:
    tool_name: str
    tool_description: str
    category: str | None
    api_scenarios: list[ToolScenario]
    provenance: str


@dataclass
class QueryRecord:
    query_id: str
    query_text: str
    gold_parent_tools: list[str]
    provenance: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate exact-cosine retrieval recall at increasing ToolBench catalog sizes. "
            "Catalogs are sampled at the tool level and expanded into API-level retrieval scenarios."
        )
    )
    parser.add_argument("--query-file", type=Path, default=DEFAULT_QUERY_FILE)
    parser.add_argument(
        "--query-format",
        type=str,
        default="auto",
        choices=("auto", "toolllama_eval", "structured_queries"),
        help="How to parse the query file.",
    )
    parser.add_argument(
        "--tool-doc-dir",
        type=Path,
        default=DEFAULT_TOOL_DOC_DIR,
        help=(
            "Optional directory containing ToolBench-style tool JSON docs. "
            "If absent, the script falls back to tool docs extracted from the query file."
        ),
    )
    parser.add_argument(
        "--tool-doc-file",
        type=Path,
        default=None,
        help="Optional single JSON file containing extra tool docs in ToolBench-like format.",
    )
    parser.add_argument("--catalog-sizes", type=str, default="32,100,250,500")
    parser.add_argument("--subset-seeds", type=str, default="1,2,3")
    parser.add_argument("--query-count", type=int, default=250)
    parser.add_argument("--query-seed", type=int, default=42)
    parser.add_argument(
        "--query-id-file",
        type=Path,
        default=None,
        help=(
            "Optional JSON or line-delimited file listing the exact held-out query IDs to evaluate. "
            "When provided, the script uses those IDs instead of random sampling."
        ),
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--embed-batch-size", type=int, default=64)
    parser.add_argument(
        "--base-tool-order",
        type=str,
        default="first_seen",
        choices=("first_seen", "alphabetical"),
        help="How to choose the fixed 32-tool base catalog.",
    )
    parser.add_argument(
        "--base-tool-manifest",
        type=Path,
        default=None,
        help=(
            "Optional JSON or line-delimited file listing the exact 32 parent tools used for the base catalog. "
            "When omitted, the script falls back to the local reconstruction."
        ),
    )
    parser.add_argument(
        "--require-validated-provenance",
        action="store_true",
        help=(
            "Fail unless both --query-id-file and --base-tool-manifest are provided. "
            "Use this for rebuttal-safe runs that must not rely on local reconstructions."
        ),
    )
    parser.add_argument(
        "--precompute-query-embeddings",
        action="store_true",
        help="Retained for CLI compatibility; the optimized runner always precomputes and amortizes query embeddings.",
    )
    parser.add_argument(
        "--allow-missing-sizes",
        action="store_true",
        help="Skip catalog sizes that exceed the local tool inventory instead of failing.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Parse inputs and build catalogs without calling Jina.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def normalize_tool_name(value: str) -> str:
    return re.sub(r"\s+", "_", value.strip().lower())


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return float("-inf")
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(a) * float(a) for a in left))
    right_norm = math.sqrt(sum(float(b) * float(b) for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return float("-inf")
    return dot / (left_norm * right_norm)


def mean_sd(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values) if values else 0.0
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, sd


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return sorted_values[lower]
    weight = index - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def batched(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_string_list(path: Path) -> list[str]:
    raw_text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(raw_text)
        if isinstance(payload, dict):
            if "query_ids" in payload and isinstance(payload["query_ids"], list):
                payload = payload["query_ids"]
            elif "tool_names" in payload and isinstance(payload["tool_names"], list):
                payload = payload["tool_names"]
        if not isinstance(payload, list):
            raise ValueError(f"Expected a JSON list in {path}")
        values = [str(item).strip() for item in payload if str(item).strip()]
    else:
        values = [line.strip() for line in raw_text.splitlines() if line.strip()]
    if not values:
        raise ValueError(f"No values found in {path}")
    return values


def select_query_records(records: list[QueryRecord], *, query_count: int, query_seed: int, query_ids: list[str] | None) -> tuple[list[QueryRecord], dict[str, Any]]:
    if query_ids:
        if query_count != len(query_ids):
            raise ValueError(
                f"--query-count={query_count} does not match the {len(query_ids)} IDs provided in the explicit query split."
            )
        by_id = {record.query_id: record for record in records}
        missing = [query_id for query_id in query_ids if query_id not in by_id]
        if missing:
            raise ValueError(
                f"{len(missing)} query IDs from the explicit split are missing in the query source. "
                f"Examples: {missing[:5]}"
            )
        return [by_id[query_id] for query_id in query_ids], {
            "selection_mode": "explicit_query_ids",
            "query_id_source": "explicit_file",
        }

    if query_count > len(records):
        raise ValueError(f"Requested {query_count} queries, but only {len(records)} parsable queries are available.")
    rng = random.Random(query_seed)
    chosen_indices = sorted(rng.sample(range(len(records)), query_count))
    return [records[idx] for idx in chosen_indices], {
        "selection_mode": "random_sample",
        "query_id_source": "sampled_from_query_source",
    }


def extract_tool_descriptions(system_prompt: str) -> dict[str, str]:
    marker = "You have access of the following tools:\n"
    end_marker = "\n\nSpecifically, you have access to the following APIs:"
    if marker not in system_prompt or end_marker not in system_prompt:
        return {}
    block = system_prompt.split(marker, 1)[1].split(end_marker, 1)[0]
    descriptions: dict[str, str] = {}
    for line in block.splitlines():
        match = re.match(r"\s*\d+\.(.+?):\s*(.+)\s*$", line)
        if match:
            descriptions[normalize_tool_name(match.group(1))] = match.group(2).strip()
    return descriptions


def parse_api_list_from_prompt(system_prompt: str) -> list[dict[str, Any]]:
    match = re.search(r"Specifically, you have access to the following APIs: (\[.*\])", system_prompt, re.S)
    if not match:
        return []
    try:
        parsed = ast.literal_eval(match.group(1))
    except Exception:
        return []
    return [item for item in parsed if isinstance(item, dict)]


def infer_parent_tool(api_name: str, description: str) -> str | None:
    match = re.search(r'tool "([^"]+)"', description)
    if match:
        return normalize_tool_name(match.group(1))
    match = re.search(r"_for_([a-zA-Z0-9_]+)$", api_name)
    if match:
        return normalize_tool_name(match.group(1))
    return None


def extract_action_name(message: str) -> str | None:
    match = re.search(r"Action:\s*([A-Za-z0-9_]+)", message)
    return match.group(1) if match else None


def clean_query_text(raw: str) -> str:
    text = raw.replace("\r\n", "\n").strip()
    return re.sub(r"\nBegin!\s*$", "", text).strip()


def parse_toolllama_eval_queries(path: Path, *, query_count: int, query_seed: int, query_ids: list[str] | None) -> tuple[list[QueryRecord], dict[str, ToolDoc], dict[str, Any]]:
    payload = load_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list in {path}")

    extracted_queries: list[QueryRecord] = []
    tool_docs: dict[str, ToolDoc] = {}
    parse_failures = 0

    for idx, row in enumerate(payload):
        if not isinstance(row, dict):
            continue
        conversations = row.get("conversations")
        if not isinstance(conversations, list) or len(conversations) < 2:
            continue
        system_prompt = str(conversations[0].get("value") or "")
        user_message = next((msg for msg in conversations if msg.get("from") == "user"), None)
        if not user_message:
            continue
        query_text = clean_query_text(str(user_message.get("value") or ""))
        if not query_text:
            continue

        tool_descriptions = extract_tool_descriptions(system_prompt)
        api_specs = parse_api_list_from_prompt(system_prompt)
        api_to_parent: dict[str, str] = {}

        for api in api_specs:
            api_name = str(api.get("name") or "").strip()
            description = str(api.get("description") or "").strip()
            if not api_name or api_name == "Finish":
                continue
            parent_tool = infer_parent_tool(api_name, description)
            if not parent_tool:
                continue
            api_to_parent[api_name] = parent_tool

            params = api.get("parameters") or {}
            properties = params.get("properties") if isinstance(params, dict) else {}
            parameter_names = [str(key) for key in properties.keys()] if isinstance(properties, dict) else []
            tool_description = tool_descriptions.get(parent_tool, "")
            scenario_text = (
                f"Tool: {parent_tool}. "
                f"Tool description: {tool_description}. "
                f"API: {api_name}. "
                f"API description: {description}. "
                f"Parameters: {', '.join(parameter_names) if parameter_names else 'none'}."
            )
            scenario = ToolScenario(
                parent_tool=parent_tool,
                scenario_id=f"{parent_tool}::{api_name}",
                scenario_name=api_name,
                text=scenario_text,
                provenance="toolllama_eval_prompt",
            )
            doc = tool_docs.setdefault(
                parent_tool,
                ToolDoc(
                    tool_name=parent_tool,
                    tool_description=tool_description,
                    category=None,
                    api_scenarios=[],
                    provenance="toolllama_eval_prompt",
                ),
            )
            if scenario.scenario_id not in {item.scenario_id for item in doc.api_scenarios}:
                doc.api_scenarios.append(scenario)

        gold_tools: list[str] = []
        for message in conversations:
            if message.get("from") != "assistant":
                continue
            action_name = extract_action_name(str(message.get("value") or ""))
            if not action_name or action_name == "Finish":
                continue
            parent_tool = api_to_parent.get(action_name)
            if parent_tool and parent_tool not in gold_tools:
                gold_tools.append(parent_tool)
        if not gold_tools:
            parse_failures += 1
            continue

        query_id = str(row.get("id") or f"toolllama_eval_{idx}")
        extracted_queries.append(QueryRecord(query_id=query_id, query_text=query_text, gold_parent_tools=gold_tools, provenance="toolllama_eval_actions"))

    sampled_queries, selection_metadata = select_query_records(
        extracted_queries,
        query_count=query_count,
        query_seed=query_seed,
        query_ids=query_ids,
    )
    metadata = {
        "query_source": str(path),
        "query_format": "toolllama_eval",
        "total_rows": len(payload),
        "parsable_queries": len(extracted_queries),
        "query_count": query_count,
        "query_seed": query_seed,
        "parse_failures": parse_failures,
        **selection_metadata,
    }
    if query_ids:
        metadata["query_id_file_note"] = "Query set comes from an explicit held-out ID file supplied at runtime."
    else:
        metadata["local_note"] = (
            "This local query set is sampled from the preserved 300-trace ToolLLaMA eval file. "
            "If you have the paper-time 250-query ToolBench split, pass that file explicitly."
        )
    return sampled_queries, tool_docs, metadata


def parse_structured_queries(path: Path, *, query_count: int, query_seed: int, query_ids: list[str] | None) -> tuple[list[QueryRecord], dict[str, Any]]:
    payload = load_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list in {path}")
    records: list[QueryRecord] = []
    for idx, row in enumerate(payload):
        if not isinstance(row, dict):
            continue
        query_text = str(row.get("query") or "").strip()
        relevant = row.get("relevant APIs") or []
        gold_tools: list[str] = []
        for item in relevant:
            if not isinstance(item, list) or len(item) < 1:
                continue
            gold = normalize_tool_name(str(item[0]))
            if gold not in gold_tools:
                gold_tools.append(gold)
        if query_text and gold_tools:
            records.append(QueryRecord(query_id=str(row.get("query_id") or f"structured_{idx}"), query_text=query_text, gold_parent_tools=gold_tools, provenance="structured_queries"))
    sampled, selection_metadata = select_query_records(
        records,
        query_count=query_count,
        query_seed=query_seed,
        query_ids=query_ids,
    )
    metadata = {
        "query_source": str(path),
        "query_format": "structured_queries",
        "total_rows": len(payload),
        "parsable_queries": len(records),
        "query_count": query_count,
        "query_seed": query_seed,
        **selection_metadata,
    }
    return sampled, metadata


def parse_tool_doc_payload(raw: Any, source_path: Path) -> list[ToolDoc]:
    docs: list[ToolDoc] = []
    if not isinstance(raw, dict):
        return docs
    tool_name = normalize_tool_name(str(raw.get("tool_name") or raw.get("standardized_name") or raw.get("title") or source_path.stem))
    api_list = raw.get("api_list")
    if not isinstance(api_list, list):
        return docs
    tool_description = str(raw.get("tool_description") or raw.get("description") or raw.get("title") or tool_name)
    category = source_path.parent.name if source_path.parent != source_path.parent.parent else None
    scenarios: list[ToolScenario] = []
    for api in api_list:
        if not isinstance(api, dict):
            continue
        api_name = str(api.get("name") or "").strip()
        description = str(api.get("description") or "").strip()
        if not api_name:
            continue
        required = [str(item) for item in api.get("required_parameters") or []]
        optional = [str(item) for item in api.get("optional_parameters") or []]
        text = (
            f"Tool: {tool_name}. "
            f"Tool description: {tool_description}. "
            f"API: {api_name}. "
            f"API description: {description}. "
            f"Required parameters: {', '.join(required) if required else 'none'}. "
            f"Optional parameters: {', '.join(optional) if optional else 'none'}."
        )
        scenarios.append(ToolScenario(parent_tool=tool_name, scenario_id=f"{tool_name}::{api_name}", scenario_name=api_name, text=text, provenance=str(source_path)))
    if scenarios:
        docs.append(ToolDoc(tool_name=tool_name, tool_description=tool_description, category=category, api_scenarios=scenarios, provenance=str(source_path)))
    return docs


def load_tool_docs_from_paths(directory: Path | None, file_path: Path | None) -> dict[str, ToolDoc]:
    docs: dict[str, ToolDoc] = {}
    candidate_files: list[Path] = []
    if directory and directory.exists():
        candidate_files.extend(sorted(directory.rglob("*.json")))
    if file_path and file_path.exists():
        candidate_files.append(file_path)
    for path in candidate_files:
        try:
            raw = load_json(path)
        except Exception:
            continue
        for doc in parse_tool_doc_payload(raw, path):
            existing = docs.get(doc.tool_name)
            if existing is None:
                docs[doc.tool_name] = doc
                continue
            seen = {item.scenario_id for item in existing.api_scenarios}
            for scenario in doc.api_scenarios:
                if scenario.scenario_id not in seen:
                    existing.api_scenarios.append(scenario)
    return docs


def merge_tool_docs(primary: dict[str, ToolDoc], fallback: dict[str, ToolDoc]) -> dict[str, ToolDoc]:
    merged: dict[str, ToolDoc] = {}
    for source in (fallback, primary):
        for tool_name, doc in source.items():
            current = merged.get(tool_name)
            if current is None:
                merged[tool_name] = ToolDoc(tool_name=doc.tool_name, tool_description=doc.tool_description, category=doc.category, api_scenarios=list(doc.api_scenarios), provenance=doc.provenance)
                continue
            if not current.tool_description and doc.tool_description:
                current.tool_description = doc.tool_description
            if not current.category and doc.category:
                current.category = doc.category
            seen = {item.scenario_id for item in current.api_scenarios}
            for scenario in doc.api_scenarios:
                if scenario.scenario_id not in seen:
                    current.api_scenarios.append(scenario)
    return merged


def ordered_tool_names(tool_docs: dict[str, ToolDoc], order: str) -> list[str]:
    names = list(tool_docs.keys())
    return sorted(names) if order == "alphabetical" else names


def select_catalog_tools(
    *,
    size: int,
    tool_order: list[str],
    query_tool_priority: list[str],
    subset_seed: int | None,
    base_tool_manifest: list[str] | None,
) -> list[str]:
    if base_tool_manifest:
        fixed_base = list(base_tool_manifest[: min(32, len(base_tool_manifest), size)])
    else:
        fixed_base = list(query_tool_priority[: min(32, len(query_tool_priority), size)])
    if size <= len(fixed_base):
        return fixed_base[:size]
    chosen = list(fixed_base)
    remaining = [tool for tool in tool_order if tool not in chosen]
    extra_needed = size - len(chosen)
    if subset_seed is None:
        chosen.extend(remaining[:extra_needed])
    else:
        rng = random.Random(subset_seed)
        chosen.extend(sorted(rng.sample(remaining, extra_needed)))
    return chosen


def flatten_scenarios(tool_names: list[str], tool_docs: dict[str, ToolDoc]) -> list[ToolScenario]:
    scenarios: list[ToolScenario] = []
    for tool_name in tool_names:
        scenarios.extend(tool_docs[tool_name].api_scenarios)
    return scenarios


def index_size_kb(scenarios: list[ToolScenario], embedding_dim: int) -> float:
    manifest = {
        "scenarios": [
            {
                "parent_tool": item.parent_tool,
                "scenario_id": item.scenario_id,
                "scenario_name": item.scenario_name,
                "text": item.text,
                "provenance": item.provenance,
            }
            for item in scenarios
        ],
        "embedding_dim": embedding_dim,
    }
    manifest_bytes = len(json.dumps(manifest, ensure_ascii=True).encode("utf-8"))
    embedding_bytes = len(scenarios) * embedding_dim * 4
    return (manifest_bytes + embedding_bytes) / 1024.0


def embed_texts(jina_client: JinaAIClient, texts: list[str], batch_size: int, *, label: str) -> tuple[list[list[float]], float]:
    vectors: list[list[float]] = []
    total_seconds = 0.0
    total_batches = max(1, math.ceil(len(texts) / batch_size)) if texts else 0
    for batch_idx, batch in enumerate(batched(texts, batch_size), start=1):
        start = time.perf_counter()
        chunk = jina_client.get_embeddings_with_retry(batch)
        total_seconds += time.perf_counter() - start
        if len(chunk) != len(batch):
            raise RuntimeError(f"Embedding mismatch for {label}: requested {len(batch)}, received {len(chunk)}")
        vectors.extend(chunk)
        print(f"[embed] {label} batch {batch_idx}/{total_batches} size={len(batch)}", flush=True)
    return vectors, total_seconds


def build_global_embedding_cache(jina_client: JinaAIClient, tool_docs: dict[str, ToolDoc], batch_size: int) -> dict[str, list[float]]:
    unique_texts: list[str] = []
    seen: set[str] = set()
    for doc in tool_docs.values():
        for scenario in doc.api_scenarios:
            if scenario.text not in seen:
                seen.add(scenario.text)
                unique_texts.append(scenario.text)
    print(f"[setup] embedding {len(unique_texts)} unique scenario texts once", flush=True)
    vectors, total_seconds = embed_texts(jina_client, unique_texts, batch_size, label="scenario-cache")
    print(f"[setup] scenario cache ready in {total_seconds:.1f}s", flush=True)
    return {text: vector for text, vector in zip(unique_texts, vectors)}


def precompute_queries(jina_client: JinaAIClient, queries: list[QueryRecord], batch_size: int) -> tuple[dict[str, list[float]], float]:
    print(f"[setup] embedding {len(queries)} queries once", flush=True)
    vectors, total_seconds = embed_texts(jina_client, [item.query_text for item in queries], batch_size, label="queries")
    per_query_seconds = total_seconds / len(queries) if queries else 0.0
    return {query.query_id: vector for query, vector in zip(queries, vectors)}, per_query_seconds


def evaluate_catalog(
    *,
    catalog_name: str,
    tool_names: list[str],
    tool_docs: dict[str, ToolDoc],
    queries: list[QueryRecord],
    top_k: int,
    scenario_cache: dict[str, list[float]],
    precomputed_queries: dict[str, list[float]],
    amortized_query_embed_ms: float,
) -> dict[str, Any]:
    scenarios = flatten_scenarios(tool_names, tool_docs)
    if not scenarios:
        raise ValueError(f"Catalog {catalog_name} contains no scenarios.")
    scenario_embeddings = [scenario_cache[item.text] for item in scenarios]
    embedding_dim = len(scenario_embeddings[0]) if scenario_embeddings else 0

    latencies_ms: list[float] = []
    search_latencies_ms: list[float] = []
    hits = 0
    rows: list[dict[str, Any]] = []

    for idx, query in enumerate(queries, start=1):
        query_embedding = precomputed_queries[query.query_id]
        start = time.perf_counter()
        scored = [
            (cosine_similarity(query_embedding, scenario_vector), scenario)
            for scenario_vector, scenario in zip(scenario_embeddings, scenarios)
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        top = scored[:top_k]
        search_latency_ms = (time.perf_counter() - start) * 1000.0
        latency_ms = search_latency_ms + amortized_query_embed_ms
        search_latencies_ms.append(search_latency_ms)
        latencies_ms.append(latency_ms)

        predicted_tools = [scenario.parent_tool for _score, scenario in top]
        matched = any(tool in query.gold_parent_tools for tool in predicted_tools)
        if matched:
            hits += 1
        rows.append(
            {
                "query_id": query.query_id,
                "query_text": query.query_text,
                "gold_parent_tools": query.gold_parent_tools,
                "top_parent_tools": predicted_tools,
                "matched": matched,
                "latency_ms": latency_ms,
                "search_latency_ms": search_latency_ms,
            }
        )
        if idx % 50 == 0 or idx == len(queries):
            print(f"[eval] {catalog_name} query {idx}/{len(queries)} hits={hits}", flush=True)

    return {
        "catalog_name": catalog_name,
        "tool_count": len(tool_names),
        "scenario_count": len(scenarios),
        "recall_at_5_fraction": f"{hits}/{len(queries)}",
        "recall_at_5": hits / len(queries) if queries else 0.0,
        "hit_count": hits,
        "query_count": len(queries),
        "latency_ms": {
            "p50": percentile(latencies_ms, 0.50),
            "p95": percentile(latencies_ms, 0.95),
            "mean": sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0,
        },
        "search_only_latency_ms": {
            "p50": percentile(search_latencies_ms, 0.50),
            "p95": percentile(search_latencies_ms, 0.95),
            "mean": sum(search_latencies_ms) / len(search_latencies_ms) if search_latencies_ms else 0.0,
        },
        "query_embedding_mode": "precomputed_amortized",
        "index_size_kb": index_size_kb(scenarios, embedding_dim),
        "tool_names": tool_names,
        "rows": rows,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# ToolBench Retrieval-at-Scale",
        "",
        f"- Query source: `{summary['query_metadata']['query_source']}`",
        f"- Query format: `{summary['query_metadata']['query_format']}`",
        f"- Query count: `{summary['query_count']}`",
        f"- Embedder: `{summary['embed_model']}`",
        f"- Search: `exact cosine`",
        f"- Top-K: `{summary['top_k']}`",
        f"- Query embeddings: `{summary['query_embedding_mode']}`",
        "",
        "| Catalog size | Recall@5 | Recall | p50 ms | p95 ms | Index KB | Notes |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary["table_rows"]:
        note = row.get("note", "")
        recall_fraction = row.get("recall_at_5_fraction", "-")
        recall_value = row.get("recall_at_5")
        p50 = row.get("p50_latency_ms")
        p95 = row.get("p95_latency_ms")
        index_kb = row.get("index_size_kb")
        if recall_value is None:
            lines.append(f"| {row['catalog_size']} | - | - | - | - | - | {note} |")
        else:
            lines.append(f"| {row['catalog_size']} | {recall_fraction} | {recall_value:.3f} | {p50:.1f} | {p95:.1f} | {index_kb:.1f} | {note} |")
    lines.extend([
        "",
        "## Provenance",
        "",
        f"- Base catalog provenance: {summary['catalog_provenance']['base_catalog']}",
        f"- Added-tool provenance: {summary['catalog_provenance']['added_tools']}",
        f"- Hardware: {summary['hardware_note']}",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    load_dotenv(REPO_ROOT / ".env")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    catalog_sizes = parse_int_list(args.catalog_sizes)
    subset_seeds = parse_int_list(args.subset_seeds)
    explicit_query_ids = load_string_list(args.query_id_file) if args.query_id_file else None
    explicit_base_tools = [normalize_tool_name(item) for item in load_string_list(args.base_tool_manifest)] if args.base_tool_manifest else None

    if args.require_validated_provenance:
        if explicit_query_ids is None:
            raise ValueError("Validated provenance mode requires --query-id-file so the held-out query split is explicit.")
        if explicit_base_tools is None:
            raise ValueError("Validated provenance mode requires --base-tool-manifest so the fixed 32-tool catalog is explicit.")

    query_format = "toolllama_eval" if args.query_format == "auto" and args.query_file.name.endswith("_eval.json") else args.query_format
    if query_format == "auto":
        query_format = "structured_queries"

    extracted_tool_docs: dict[str, ToolDoc] = {}
    if query_format == "toolllama_eval":
        queries, extracted_tool_docs, query_metadata = parse_toolllama_eval_queries(
            args.query_file,
            query_count=args.query_count,
            query_seed=args.query_seed,
            query_ids=explicit_query_ids,
        )
    else:
        queries, query_metadata = parse_structured_queries(
            args.query_file,
            query_count=args.query_count,
            query_seed=args.query_seed,
            query_ids=explicit_query_ids,
        )

    local_tool_docs = load_tool_docs_from_paths(args.tool_doc_dir, args.tool_doc_file)
    tool_docs = merge_tool_docs(primary=local_tool_docs, fallback=extracted_tool_docs)
    if not tool_docs:
        raise ValueError("No tool docs were found. Pass --tool-doc-dir or --tool-doc-file, or use the ToolLLaMA eval file.")

    tool_order = ordered_tool_names(tool_docs, args.base_tool_order)
    query_tool_priority: list[str] = []
    for query in queries:
        for tool_name in query.gold_parent_tools:
            if tool_name in tool_docs and tool_name not in query_tool_priority:
                query_tool_priority.append(tool_name)
    for tool_name in tool_order:
        if tool_name not in query_tool_priority:
            query_tool_priority.append(tool_name)

    if explicit_base_tools:
        missing_tools = [tool for tool in explicit_base_tools if tool not in tool_docs]
        if missing_tools:
            raise ValueError(
                f"{len(missing_tools)} tools from the explicit 32-tool manifest are missing in the local tool docs. "
                f"Examples: {missing_tools[:5]}"
            )
        if len(explicit_base_tools) != 32:
            raise ValueError(
                f"Expected the explicit base manifest to contain 32 tools, but found {len(explicit_base_tools)}."
            )

    max_inventory = len(tool_order)
    requested_max = max(catalog_sizes) if catalog_sizes else 0
    if requested_max > max_inventory and not args.allow_missing_sizes:
        raise ValueError(
            f"Requested catalog size {requested_max}, but only {max_inventory} tools are available locally. "
            "Provide a larger tool dump or rerun with --allow-missing-sizes."
        )

    dry_run_summary = {
        "query_count": len(queries),
        "unique_gold_tools_in_queries": len({tool for query in queries for tool in query.gold_parent_tools}),
        "available_tool_docs": len(tool_docs),
        "requested_catalog_sizes": catalog_sizes,
        "max_local_inventory": max_inventory,
        "query_metadata": query_metadata,
        "catalog_provenance": {
            "base_catalog": (
                f"Fixed 32-tool base loaded from `{args.base_tool_manifest}`."
                if args.base_tool_manifest
                else "Fixed 32-tool base assembled from the earliest locally observed query-relevant tools, "
                "because the paper-time 32-tool manifest is not preserved in this repo."
            ),
            "added_tools": (
                f"Local tool inventory merged from `{args.query_file}` and `{args.tool_doc_dir}`"
                if args.tool_doc_dir.exists()
                else f"Local tool inventory extracted from `{args.query_file}`"
            ),
        },
    }

    if args.dry_run:
        output_path = args.output_dir / "dry_run_summary.json"
        output_path.write_text(json.dumps(dry_run_summary, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {os.path.relpath(output_path, REPO_ROOT)}")
        return

    jina_client = JinaAIClient()
    embed_model = os.environ.get("JINA_EMBED_MODEL", "jina-embeddings-v2-base-en")
    scenario_cache = build_global_embedding_cache(jina_client, tool_docs, args.embed_batch_size)
    precomputed_queries, amortized_query_seconds = precompute_queries(jina_client, queries, args.embed_batch_size)
    amortized_query_ms = amortized_query_seconds * 1000.0

    evaluated_rows: list[dict[str, Any]] = []
    run_details: list[dict[str, Any]] = []

    for size in catalog_sizes:
        if size > max_inventory:
            if args.allow_missing_sizes:
                evaluated_rows.append({"catalog_size": size, "note": f"skipped: local inventory has only {max_inventory} tools"})
                continue
            raise ValueError(f"Catalog size {size} exceeds local inventory of {max_inventory} tools.")

        seeds_to_run = subset_seeds if size in {100, 250} else [None]
        seed_results: list[dict[str, Any]] = []
        print(f"[catalog] size={size} seeds={seeds_to_run}", flush=True)
        for subset_seed in seeds_to_run:
            tool_names = select_catalog_tools(
                size=size,
                tool_order=tool_order,
                query_tool_priority=query_tool_priority,
                subset_seed=subset_seed,
                base_tool_manifest=explicit_base_tools,
            )
            catalog_name = f"catalog_{size}" if subset_seed is None else f"catalog_{size}_seed_{subset_seed}"
            result = evaluate_catalog(
                catalog_name=catalog_name,
                tool_names=tool_names,
                tool_docs=tool_docs,
                queries=queries,
                top_k=args.top_k,
                scenario_cache=scenario_cache,
                precomputed_queries=precomputed_queries,
                amortized_query_embed_ms=amortized_query_ms,
            )
            result["subset_seed"] = subset_seed
            seed_results.append(result)

        recall_values = [item["recall_at_5"] for item in seed_results]
        p50_values = [item["latency_ms"]["p50"] for item in seed_results]
        p95_values = [item["latency_ms"]["p95"] for item in seed_results]
        index_values = [item["index_size_kb"] for item in seed_results]
        mean_recall, sd_recall = mean_sd(recall_values)
        mean_p50, _ = mean_sd(p50_values)
        mean_p95, _ = mean_sd(p95_values)
        mean_index, _ = mean_sd(index_values)

        fraction_note = (
            ", ".join(f"seed {item['subset_seed']}: {item['recall_at_5_fraction']}" for item in seed_results)
            if len(seed_results) > 1
            else seed_results[0]["recall_at_5_fraction"]
        )
        row: dict[str, Any] = {
            "catalog_size": size,
            "recall_at_5_fraction": seed_results[0]["recall_at_5_fraction"] if len(seed_results) == 1 else fraction_note,
            "recall_at_5": mean_recall,
            "sd_recall_at_5": sd_recall,
            "p50_latency_ms": mean_p50,
            "p95_latency_ms": mean_p95,
            "index_size_kb": mean_index,
            "query_embedding_mode": "precomputed_amortized",
            "note": "",
        }
        if len(seed_results) > 1:
            row["note"] = f"mean+-sd recall = {mean_recall:.3f} +- {sd_recall:.3f}"
        if size == 32 and mean_recall < 0.95:
            row["note"] = (row["note"] + "; " if row["note"] else "") + "red flag: 32-tool recall < 0.95"

        evaluated_rows.append(row)
        run_details.append(
            {
                "catalog_size": size,
                "seed_results": seed_results,
                "mean_recall_at_5": mean_recall,
                "sd_recall_at_5": sd_recall,
                "mean_p50_latency_ms": mean_p50,
                "mean_p95_latency_ms": mean_p95,
                "mean_index_size_kb": mean_index,
            }
        )

    summary = {
        "query_count": len(queries),
        "top_k": args.top_k,
        "embed_model": embed_model,
        "query_embedding_mode": "precomputed_amortized",
        "query_metadata": query_metadata,
        "catalog_provenance": dry_run_summary["catalog_provenance"],
        "hardware_note": "Runs on the local machine using Jina embeddings and exact cosine search in Python.",
        "table_rows": evaluated_rows,
        "run_details": run_details,
        "amortized_query_embedding_ms": amortized_query_ms,
    }

    summary_path = args.output_dir / "summary.json"
    markdown_path = args.output_dir / "summary.md"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(summary), encoding="utf-8")
    print(f"Wrote {os.path.relpath(summary_path, REPO_ROOT)}")
    print(f"Wrote {os.path.relpath(markdown_path, REPO_ROOT)}")


if __name__ == "__main__":
    main()
