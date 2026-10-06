#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import statistics
import subprocess
import sys
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from math_qa import JinaAIClient
from scripts.model_paths import default_llama31_8b_path
from scripts.run_gsm8k_small_router_sweep import cosine_similarity, parse_seed_list
from scripts.run_reranker_prompt_sweep import (
    RoutedCandidate,
    build_query_embeddings,
    load_local_backend,
    render_precautions,
    run_prompt,
)
from synapse.edge.aggregator import EdgeAggregator, EdgeConfig
from synapse.knowledge.compendium import KnowledgeArtifact, KnowledgePackage

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.svm import LinearSVC
except Exception:
    TfidfVectorizer = None
    OneVsRestClassifier = None
    MultiLabelBinarizer = None
    LinearSVC = None

DEFAULT_MODEL_PATH = default_llama31_8b_path()
DEFAULT_STB_ROOT = REPO_ROOT / "external_datasets" / "StableToolBench"
DEFAULT_TOOLBENCH_INSTRUCTION_DIR = REPO_ROOT / "external_datasets" / "toolbench_hf" / "instruction"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "verification" / "stabletoolbench_federated"
DEFAULT_LOCAL_JINA_MODEL = REPO_ROOT / "external_models" / "jinaai" / "jina-embeddings-v2-base-en"
DEFAULT_LOCAL_JINA_HF_REF = REPO_ROOT / "external_models" / "hf_home" / "hub" / "models--jinaai--jina-embeddings-v2-base-en" / "refs" / "main"
GROUPS = ["G1_instruction", "G1_tool", "G1_category", "G2_instruction", "G2_category", "G3_instruction"]


@dataclass
class TestQuery:
    query_id: str
    query: str
    gold_tools: list[str]
    group: str
    categories: list[str]
    api_list: list[dict[str, Any]]


@dataclass
class ExperienceItem:
    query_id: str
    query: str
    gold_tools: list[str]
    group: str
    categories: list[str]
    primary_category: str
    api_list: list[dict[str, Any]]


@dataclass
class ToolDoc:
    tool_name: str
    categories: list[str]
    api_names: list[str]
    descriptions: list[str]


@dataclass
class ClassifierBundle:
    vectorizer: Any
    classifier: Any
    classes: list[str]


class ProgressLogger:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.progress_path = output_dir / "progress.jsonl"
        self.status_path = output_dir / "status.json"
        self.started_at = time.time()

    def log(self, stage: str, **payload: Any) -> None:
        event = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "elapsed_seconds": round(time.time() - self.started_at, 3),
            "stage": stage,
            **payload,
        }
        line = json.dumps(event, ensure_ascii=True, sort_keys=True)
        print(line, flush=True)
        with self.progress_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        self.status_path.write_text(json.dumps(event, indent=2), encoding="utf-8")


@contextmanager
def merge_heartbeat(progress: ProgressLogger, *, seed: int, arm: str, merge_policy: str, interval_s: float = 30.0):
    stop_event = threading.Event()
    started_at = time.time()

    def _worker() -> None:
        beat = 0
        while not stop_event.wait(interval_s):
            beat += 1
            progress.log(
                "arm_merge_heartbeat",
                seed=seed,
                arm=arm,
                merge_policy=merge_policy,
                heartbeat_idx=beat,
                merge_elapsed_s=round(time.time() - started_at, 3),
            )

    thread = threading.Thread(target=_worker, name=f"merge-heartbeat-{seed}-{arm}", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        thread.join(timeout=1.0)


@contextmanager
def temporary_env(updates: dict[str, str | None]):
    previous = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Federated StableToolBench runner.")
    parser.add_argument("--stb-root", type=Path, default=DEFAULT_STB_ROOT)
    parser.add_argument("--toolbench-instruction-dir", type=Path, default=DEFAULT_TOOLBENCH_INSTRUCTION_DIR)
    parser.add_argument("--groups", type=str, default=",".join(GROUPS))
    parser.add_argument("--data-mode", type=str, required=True, choices=["stable_holdout", "toolbench_train"])
    parser.add_argument("--test-fraction", type=float, default=0.4)
    parser.add_argument("--client-count", type=int, default=5)
    parser.add_argument("--partition-mode", type=str, default="category", choices=["category", "iid"])
    parser.add_argument("--arms", type=str, default="synapse,centralized,local_only,flat_pool,query_classifier")
    parser.add_argument("--merge-policy", type=str, default="conflict_log", choices=["conflict_log", "round_delayed", "majority"])
    parser.add_argument("--seeds", type=str, default="42,123,456,789,1024")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retrieval-pool-size", type=int, default=5)
    parser.add_argument("--retrieval-mode", type=str, default="entry_topk", choices=["entry_topk", "distinct_tool_topk", "union_doc_scenario"])
    parser.add_argument("--reranker-variant", type=str, default="V1", choices=["V1", "V3"])
    parser.add_argument("--doc-drift-fraction", type=float, default=0.0)
    parser.add_argument("--doc-drift-mode", type=str, default="none", choices=["none", "stale", "truncated", "renamed"])
    parser.add_argument("--doc-drift-seed-offset", type=int, default=1000)
    parser.add_argument("--max-train-items", type=int, default=0)
    parser.add_argument("--max-items-per-client", type=int, default=0)
    parser.add_argument("--junk-filter", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--skip-package-persistence", action="store_true")
    parser.add_argument("--load-packages-roots", type=str, default="")
    parser.add_argument("--embed-model", type=str, default="jina-embeddings-v2-base-en")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--pool-embedding-cache-dir", type=Path, default=None)
    parser.add_argument("--contamination-near-duplicate-threshold", type=float, default=0.95)
    parser.add_argument("--contamination-report-threshold", type=float, default=0.90)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def normalize_tool_name(value: str) -> str:
    return "_".join(value.strip().lower().split())


def normalize_category(value: str) -> str:
    return "_".join(value.strip().lower().split())


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")).hexdigest()


def repo_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def repo_dirty_entries() -> list[str]:
    pathspecs = ["launch.sh", "scripts", "synapse", "math_qa.py"]
    for name in ("pyproject.toml", "requirements.txt", "requirements-dev.txt", "setup.py"):
        if (REPO_ROOT / name).exists():
            pathspecs.append(name)
    cmd = ["git", "status", "--short", "--untracked-files=all", "--", *pathspecs]
    output = subprocess.check_output(cmd, cwd=REPO_ROOT, text=True)
    return [line for line in output.splitlines() if line.strip()]


def assert_clean_tree(*, allow_dirty: bool) -> tuple[str, list[str]]:
    commit = repo_commit()
    dirty = repo_dirty_entries()
    if dirty and not allow_dirty:
        sample = "\n".join(dirty[:20])
        raise RuntimeError(
            "Refusing to run on a dirty worktree. Commit or stash code changes first, "
            "or pass --allow-dirty to override. Sample entries:\n" + sample
        )
    return commit, dirty


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def resolve_local_embedder() -> dict[str, str]:
    model_path = Path(os.environ.get("JINA_LOCAL_EMBED_MODEL") or DEFAULT_LOCAL_JINA_MODEL)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Local Jina embedder not found at {model_path}. Set JINA_LOCAL_EMBED_MODEL before launching."
        )
    revision = "unknown"
    if DEFAULT_LOCAL_JINA_HF_REF.exists():
        revision = DEFAULT_LOCAL_JINA_HF_REF.read_text(encoding="utf-8").strip() or "unknown"
    return {
        "model_path": str(model_path),
        "revision": revision,
        "device": os.environ.get("JINA_LOCAL_EMBED_DEVICE", "cuda"),
        "local_only": "1",
    }


def short_text(value: str, limit: int = 220) -> str:
    text = " ".join((value or "").split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def extract_gold_tools(record: dict[str, Any]) -> list[str]:
    tools: list[str] = []
    for pair in record.get("relevant APIs") or []:
        if isinstance(pair, list) and pair:
            tool = normalize_tool_name(str(pair[0]).strip())
            if tool and tool not in tools:
                tools.append(tool)
    return tools


def extract_categories(record: dict[str, Any]) -> list[str]:
    categories: list[str] = []
    for api in record.get("api_list") or []:
        category = normalize_category(str(api.get("category_name") or "unknown"))
        if category and category not in categories:
            categories.append(category)
    return categories or ["unknown"]


def load_stabletoolbench_queries(stb_root: Path, groups: list[str]) -> list[TestQuery]:
    query_dir = stb_root / "solvable_queries" / "test_instruction"
    queries: list[TestQuery] = []
    for group in groups:
        path = query_dir / f"{group}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        for record in data:
            query = str(record.get("query") or "").strip()
            gold_tools = extract_gold_tools(record)
            if not query or not gold_tools:
                continue
            queries.append(
                TestQuery(
                    query_id=str(record.get("query_id") or f"{group}:{stable_hash(query)[:12]}"),
                    query=query,
                    gold_tools=gold_tools,
                    group=group,
                    categories=extract_categories(record),
                    api_list=list(record.get("api_list") or []),
                )
            )
    return queries


def load_toolbench_training_items(instruction_dir: Path) -> list[ExperienceItem]:
    paths = [instruction_dir / "G1_query.json", instruction_dir / "G2_query.json", instruction_dir / "G3_query.json"]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing ToolBench training instruction files: " + ", ".join(str(path) for path in missing))
    items: list[ExperienceItem] = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        group = path.stem.replace("_query", "")
        for record in data:
            query = str(record.get("query") or "").strip()
            gold_tools = extract_gold_tools(record)
            if not query or not gold_tools:
                continue
            categories = extract_categories(record)
            items.append(
                ExperienceItem(
                    query_id=str(record.get("query_id") or record.get("id") or f"{group}:{stable_hash(query)[:12]}"),
                    query=query,
                    gold_tools=gold_tools,
                    group=group,
                    categories=categories,
                    primary_category=categories[0],
                    api_list=list(record.get("api_list") or []),
                )
            )
    return items


def split_experience_test(queries: list[TestQuery], test_fraction: float, seed: int) -> tuple[list[ExperienceItem], list[TestQuery]]:
    rng = random.Random(seed)
    train: list[ExperienceItem] = []
    test: list[TestQuery] = []
    buckets: dict[tuple[str, str], list[TestQuery]] = defaultdict(list)
    for query in queries:
        buckets[(query.group, query.categories[0])].append(query)
    for bucket in buckets.values():
        bucket = list(bucket)
        rng.shuffle(bucket)
        if len(bucket) == 1:
            item = bucket[0]
            train.append(ExperienceItem(query_id=item.query_id, query=item.query, gold_tools=item.gold_tools, group=item.group, categories=item.categories, primary_category=item.categories[0], api_list=item.api_list))
            continue
        test_count = max(1, int(round(len(bucket) * test_fraction)))
        test_split = bucket[:test_count]
        train_split = bucket[test_count:]
        if not train_split:
            train_split = test_split[:1]
            test_split = test_split[1:]
        test.extend(test_split)
        for item in train_split:
            train.append(ExperienceItem(query_id=item.query_id, query=item.query, gold_tools=item.gold_tools, group=item.group, categories=item.categories, primary_category=item.categories[0], api_list=item.api_list))
    return train, test


def build_tool_registry(items: list[TestQuery | ExperienceItem]) -> dict[str, ToolDoc]:
    registry: dict[str, ToolDoc] = {}
    for item in items:
        for api in item.api_list:
            raw_name = str(api.get("tool_name") or "").strip()
            if not raw_name:
                continue
            tool = normalize_tool_name(raw_name)
            doc = registry.setdefault(tool, ToolDoc(tool_name=tool, categories=[], api_names=[], descriptions=[]))
            category = normalize_category(str(api.get("category_name") or "unknown"))
            if category not in doc.categories:
                doc.categories.append(category)
            api_name = str(api.get("api_name") or "").strip()
            if api_name and api_name not in doc.api_names:
                doc.api_names.append(api_name)
            desc = short_text(str(api.get("api_description") or "").strip(), 280)
            if desc and desc not in doc.descriptions:
                doc.descriptions.append(desc)
    return registry


def is_junk_tool(doc: ToolDoc) -> bool:
    name = doc.tool_name.strip().lower()
    if not doc.descriptions or not any(part.strip() for part in doc.descriptions):
        return True
    if re.search(r'(^|[_-])(test|demo|asdf|placeholder)([_-]|$)', name):
        return True
    if not any(ch.isascii() and ch.isalnum() for ch in name):
        return True
    return False


def apply_junk_filter(registry: dict[str, ToolDoc]) -> tuple[dict[str, ToolDoc], dict[str, Any]]:
    kept: dict[str, ToolDoc] = {}
    dropped: list[str] = []
    for tool, doc in registry.items():
        if is_junk_tool(doc):
            dropped.append(tool)
        else:
            kept[tool] = doc
    return kept, {
        "enabled": True,
        "dropped_tool_count": len(dropped),
        "dropped_tools_sample": sorted(dropped)[:20],
    }


def filter_eval_queries(queries: list[TestQuery], registry: dict[str, ToolDoc]) -> tuple[list[TestQuery], dict[str, Any]]:
    kept: list[TestQuery] = []
    dropped_empty = 0
    pruned_gold_total = 0
    for item in queries:
        gold = [tool for tool in item.gold_tools if tool in registry]
        pruned_gold_total += len(item.gold_tools) - len(gold)
        if not gold:
            dropped_empty += 1
            continue
        kept.append(TestQuery(query_id=item.query_id, query=item.query, gold_tools=gold, group=item.group, categories=item.categories, api_list=item.api_list))
    return kept, {
        "dropped_query_count": dropped_empty,
        "pruned_gold_tool_refs": pruned_gold_total,
    }


def filter_experience_items(items: list[ExperienceItem], registry: dict[str, ToolDoc]) -> tuple[list[ExperienceItem], dict[str, Any]]:
    kept: list[ExperienceItem] = []
    dropped_empty = 0
    pruned_gold_total = 0
    for item in items:
        gold = [tool for tool in item.gold_tools if tool in registry]
        pruned_gold_total += len(item.gold_tools) - len(gold)
        if not gold:
            dropped_empty += 1
            continue
        kept.append(ExperienceItem(query_id=item.query_id, query=item.query, gold_tools=gold, group=item.group, categories=item.categories, primary_category=item.primary_category, api_list=item.api_list))
    return kept, {
        "dropped_train_item_count": dropped_empty,
        "pruned_train_gold_tool_refs": pruned_gold_total,
    }


def normalize_query_text(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())


def remove_exact_eval_overlaps(items: list[ExperienceItem], eval_queries: list[TestQuery]) -> tuple[list[ExperienceItem], dict[str, Any]]:
    eval_by_text: dict[str, list[str]] = defaultdict(list)
    for query in eval_queries:
        eval_by_text[normalize_query_text(query.query)].append(query.query_id)
    kept: list[ExperienceItem] = []
    removed: list[ExperienceItem] = []
    removed_texts: set[str] = set()
    matched_eval_ids: set[str] = set()
    for item in items:
        key = normalize_query_text(item.query)
        if key in eval_by_text:
            removed.append(item)
            matched_eval_ids.update(eval_by_text[key])
        else:
            kept.append(item)
    return kept, {
        "pool_items_removed_exact": len(removed),
        "eval_queries_with_exact_match_in_pool": len(matched_eval_ids),
        "pool_items_removed_exact_query_ids_sample": [item.query_id for item in removed[:20]],
        "eval_queries_with_exact_match_in_pool_ids_sample": sorted(matched_eval_ids)[:20],
        "eval_query_count": len(eval_queries),
    }


def _normalize_embedding_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def remove_near_duplicate_eval_overlaps(
    items: list[ExperienceItem],
    eval_queries: list[TestQuery],
    jina_client: JinaAIClient,
    embed_model: str,
    *,
    removal_threshold: float = 0.95,
    report_threshold: float = 0.90,
    batch_size: int = 1024,
    pool_embedding_cache_dir: Path | None = None,
    progress: ProgressLogger | None = None,
) -> tuple[list[ExperienceItem], dict[str, Any], set[str]]:
    if not items or not eval_queries:
        return items, {
            "near_duplicate_threshold": removal_threshold,
            "report_threshold": report_threshold,
            "pool_items_removed_near_dup": 0,
            "eval_queries_with_near_dup_in_pool": 0,
            "pool_items_removed_near_dup_query_ids_sample": [],
            "eval_queries_with_near_dup_in_pool_ids_sample": [],
            "near_duplicate_removed_count_at_report_threshold": 0,
            "near_duplicate_removed_count_if_threshold_0_90": 0,
        }, set()
    eval_embeddings = np.asarray(batched_query_embeddings(jina_client, [item.query for item in eval_queries], embed_model), dtype=np.float32)
    eval_embeddings = _normalize_embedding_matrix(eval_embeddings)
    kept: list[ExperienceItem] = []
    removed: list[ExperienceItem] = []
    removed_texts: set[str] = set()
    matched_eval_ids: set[str] = set()
    report_count = 0
    report_count_090 = 0
    samples: list[dict[str, Any]] = []
    batch_total = max(1, (len(items) + batch_size - 1) // batch_size)
    cache_info: dict[str, Any] = {
        "enabled": bool(pool_embedding_cache_dir),
        "cache_hit": False,
    }
    pool_embeddings: np.ndarray | None = None
    cache_path: Path | None = None
    if pool_embedding_cache_dir is not None:
        pool_hash = stable_hash(
            [
                {
                    "query_id": item.query_id,
                    "query": normalize_query_text(item.query),
                }
                for item in items
            ]
        )
        cache_dir = Path(pool_embedding_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"pool_emb_{embed_model}_{pool_hash}.npy"
        cache_info["cache_path"] = str(cache_path)
        cache_info["pool_hash"] = pool_hash
        if cache_path.exists():
            pool_embeddings = np.load(cache_path)
            cache_info["cache_hit"] = True
            cache_info["cached_item_count"] = int(pool_embeddings.shape[0])
    if progress is not None:
        progress.log(
            "contamination_near_duplicate_begin",
            train_count=len(items),
            eval_count=len(eval_queries),
            batch_size=batch_size,
            batch_total=batch_total,
            removal_threshold=removal_threshold,
            report_threshold=report_threshold,
            pool_embedding_cache=cache_info,
        )
    if pool_embeddings is None:
        collected_batches: list[np.ndarray] = []
        for start in range(0, len(items), batch_size):
            batch = items[start:start + batch_size]
            batch_embeddings = np.asarray(batched_query_embeddings(jina_client, [item.query for item in batch], embed_model), dtype=np.float32)
            batch_embeddings = _normalize_embedding_matrix(batch_embeddings)
            collected_batches.append(batch_embeddings)
            if progress is not None:
                progress.log(
                    "contamination_pool_embedding_batch_done",
                    batch_index=(start // batch_size) + 1,
                    batch_total=batch_total,
                    processed_count=min(start + len(batch), len(items)),
                )
        pool_embeddings = np.concatenate(collected_batches, axis=0) if collected_batches else np.zeros((0, eval_embeddings.shape[1]), dtype=np.float32)
        if cache_path is not None:
            np.save(cache_path, pool_embeddings)
            cache_info["cache_saved"] = True
            cache_info["cached_item_count"] = int(pool_embeddings.shape[0])
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        batch_embeddings = pool_embeddings[start:start + len(batch)]
        sims = batch_embeddings @ eval_embeddings.T
        best_idx = np.argmax(sims, axis=1)
        best_scores = sims[np.arange(len(batch)), best_idx]
        for item, score, idx in zip(batch, best_scores.tolist(), best_idx.tolist()):
            if score >= report_threshold:
                report_count += 1
            if score >= 0.90:
                report_count_090 += 1
            if score >= removal_threshold:
                removed.append(item)
                removed_texts.add(normalize_query_text(item.query))
                matched_eval_ids.add(eval_queries[idx].query_id)
                if len(samples) < 20:
                    samples.append({
                        "train_query_id": item.query_id,
                        "eval_query_id": eval_queries[idx].query_id,
                        "score": round(float(score), 6),
                    })
            else:
                kept.append(item)
        if progress is not None:
            progress.log(
                "contamination_near_duplicate_batch_done",
                batch_index=(start // batch_size) + 1,
                batch_total=batch_total,
                processed_count=min(start + len(batch), len(items)),
                kept_count=len(kept),
                removed_count=len(removed),
                report_count=report_count,
                report_count_090=report_count_090,
            )
    return kept, {
        "near_duplicate_threshold": removal_threshold,
        "report_threshold": report_threshold,
        "pool_items_removed_near_dup": len(removed),
        "eval_queries_with_near_dup_in_pool": len(matched_eval_ids),
        "pool_items_removed_near_dup_query_ids": [item.query_id for item in removed],
        "pool_items_removed_near_dup_query_ids_sample": [item.query_id for item in removed[:20]],
        "eval_queries_with_near_dup_in_pool_ids_sample": sorted(matched_eval_ids)[:20],
        "near_duplicate_removed_count_at_report_threshold": report_count,
        "near_duplicate_removed_count_if_threshold_0_90": report_count_090,
        "near_duplicate_removed_pairs_sample": samples,
        "pool_embedding_cache": cache_info,
    }, removed_texts


def assert_no_eval_overlap(
    items: list[ExperienceItem],
    eval_queries: list[TestQuery],
    *,
    forbidden_near_duplicate_texts: set[str],
    stage: str,
) -> dict[str, Any]:
    eval_texts = {normalize_query_text(query.query) for query in eval_queries}
    exact_hits = [item.query_id for item in items if normalize_query_text(item.query) in eval_texts]
    near_hits = [item.query_id for item in items if normalize_query_text(item.query) in forbidden_near_duplicate_texts]
    if exact_hits or near_hits:
        raise RuntimeError(
            f"Contaminated pool at {stage}: exact_overlap_count={len(exact_hits)}, near_duplicate_count={len(near_hits)}"
        )
    return {
        "stage": stage,
        "exact_overlap_count": 0,
        "near_duplicate_count": 0,
        "checked_item_count": len(items),
    }


def clone_tool_doc(doc: ToolDoc) -> ToolDoc:
    return ToolDoc(
        tool_name=doc.tool_name,
        categories=list(doc.categories),
        api_names=list(doc.api_names),
        descriptions=list(doc.descriptions),
    )


def apply_doc_drift(
    registry: dict[str, ToolDoc],
    *,
    fraction: float,
    mode: str,
    seed: int,
) -> tuple[dict[str, ToolDoc], dict[str, Any]]:
    if mode == "none" or fraction <= 0.0:
        cloned = {tool: clone_tool_doc(doc) for tool, doc in registry.items()}
        return cloned, {"enabled": False, "mode": "none", "fraction": 0.0, "drifted_tools": []}

    rng = random.Random(seed)
    cloned = {tool: clone_tool_doc(doc) for tool, doc in registry.items()}
    tools = sorted(cloned)
    drift_count = max(1, int(round(len(tools) * fraction)))
    drifted_tools = sorted(rng.sample(tools, min(drift_count, len(tools))))

    by_category: dict[str, list[str]] = defaultdict(list)
    for tool, doc in cloned.items():
        category = doc.categories[0] if doc.categories else "unknown"
        by_category[category].append(tool)

    for tool in drifted_tools:
        doc = cloned[tool]
        if mode == "truncated":
            if doc.descriptions:
                doc.descriptions = [doc.descriptions[0]]
        elif mode == "renamed":
            opaque = stable_hash({"tool": tool, "seed": seed})[:12]
            count = max(1, len(doc.api_names))
            doc.api_names = [f"endpoint_{opaque}_{idx + 1}" for idx in range(count)]
            doc.descriptions = [f"opaque_tool_id={opaque}"]
        elif mode == "stale":
            category = doc.categories[0] if doc.categories else "unknown"
            siblings = [candidate for candidate in by_category.get(category, []) if candidate != tool]
            if siblings:
                donor = cloned[rng.choice(siblings)]
                if donor.descriptions:
                    doc.descriptions = list(donor.descriptions)
                if donor.api_names:
                    doc.api_names = list(donor.api_names)

    return cloned, {
        "enabled": True,
        "mode": mode,
        "fraction": fraction,
        "drifted_tools": drifted_tools,
    }


def assign_clients(items: list[ExperienceItem], client_count: int, partition_mode: str, seed: int) -> dict[str, list[ExperienceItem]]:
    rng = random.Random(seed)
    clients = {f"client_{idx}": [] for idx in range(client_count)}
    if partition_mode == "iid":
        shuffled = list(items)
        rng.shuffle(shuffled)
        for idx, item in enumerate(shuffled):
            clients[f"client_{idx % client_count}"].append(item)
        return clients
    categories = sorted({item.primary_category for item in items})
    rng.shuffle(categories)
    category_to_client = {category: f"client_{idx % client_count}" for idx, category in enumerate(categories)}
    for item in items:
        clients[category_to_client[item.primary_category]].append(item)
    return clients


def limit_experience_items(items: list[ExperienceItem], max_items: int, seed: int) -> list[ExperienceItem]:
    if max_items <= 0 or len(items) <= max_items:
        return list(items)
    rng = random.Random(seed)
    sampled = list(items)
    rng.shuffle(sampled)
    return sampled[:max_items]


def limit_client_items(clients: dict[str, list[ExperienceItem]], max_items_per_client: int, seed: int) -> dict[str, list[ExperienceItem]]:
    if max_items_per_client <= 0:
        return clients
    limited: dict[str, list[ExperienceItem]] = {}
    for offset, client_id in enumerate(sorted(clients)):
        limited[client_id] = limit_experience_items(clients[client_id], max_items_per_client, seed + offset + 1)
    return limited


def tool_description(doc: ToolDoc) -> str:
    category = doc.categories[0] if doc.categories else "unknown"
    apis = ", ".join(doc.api_names[:3]) or doc.tool_name
    desc = "; ".join(doc.descriptions[:2])
    return f"category={category}; apis={apis}; docs={desc}" if desc else f"category={category}; apis={apis}"


def render_usage_scenario_text(tool: str, payload: dict[str, Any]) -> str:
    lines = [f"scenario: {tool}"]
    if payload.get("tool_description"):
        lines.append(f"tool_description: {payload['tool_description']}")
    if payload.get("scenario_context"):
        lines.append(f"scenario_context: {payload['scenario_context']}")
    precautions = payload.get("precautions") or []
    if precautions:
        lines.append("precautions: " + "; ".join(precautions))
    if payload.get("annex_summary"):
        lines.append(f"structured_annex: {payload['annex_summary']}")
    return "\n".join(lines)


def render_tool_doc_text(tool: str, doc: ToolDoc) -> str:
    category = ", ".join(doc.categories[:2]) if doc.categories else "unknown"
    apis = ", ".join(doc.api_names[:4]) or tool
    descriptions = "; ".join(doc.descriptions[:2])
    lines = [
        f"tool: {tool}",
        f"category: {category}",
        f"apis: {apis}",
    ]
    if descriptions:
        lines.append(f"documentation: {descriptions}")
    return "\n".join(lines)


def build_doc_package(registry: dict[str, ToolDoc], source_id: str = "tool_docs") -> tuple[KnowledgePackage, str]:
    artifacts: list[KnowledgeArtifact] = []
    for tool, doc in sorted(registry.items()):
        payload = {
            "type": "tool_doc",
            "payload_mode": "typed",
            "tool_description": tool_description(doc),
            "scenario_context": short_text("; ".join(doc.descriptions[:2]), 260),
            "precautions": [],
            "annex_summary": f"doc_source=registry; categories={','.join(doc.categories[:2]) or 'unknown'}",
        }
        metadata = {
            "tool": tool,
            "domain": tool,
            "scenario": f"{tool}__doc",
            "category": doc.categories[0] if doc.categories else "unknown",
            "source_group": "registry",
            "artifact_origin": "tool_doc",
        }
        artifacts.append(
            KnowledgeArtifact(
                signature=stable_hash({"tool_doc": tool}),
                text=render_tool_doc_text(tool, doc),
                structured_payload=payload,
                metadata=metadata,
            )
        )
    package = KnowledgePackage(source_id=source_id, artifacts=artifacts, metadata={"kind": "tool_docs"})
    return package, stable_hash([
        {"signature": artifact.signature, "tool": artifact.metadata.get("tool"), "text": artifact.text, "payload": artifact.structured_payload}
        for artifact in artifacts
    ])


def batched_query_embeddings(jina_client: JinaAIClient, texts: list[str], embed_model: str, batch_size: int = 256, max_retries: int = 4) -> list[list[float]]:
    if not texts:
        return []
    unique_texts = list(dict.fromkeys(texts))
    cache: dict[str, list[float]] = {}
    prev = os.environ.get("JINA_EMBED_MODEL")
    os.environ["JINA_EMBED_MODEL"] = embed_model
    try:
        for start in range(0, len(unique_texts), batch_size):
            batch = unique_texts[start : start + batch_size]
            delay = 1.0
            last_error: Exception | None = None
            for _attempt in range(max_retries):
                try:
                    embeddings = jina_client.get_embeddings(batch)
                    for item_text, embedding in zip(batch, embeddings):
                        cache[item_text] = embedding
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    time.sleep(delay)
                    delay *= 2.0
            if last_error is not None:
                raise last_error
    finally:
        if prev is None:
            os.environ.pop("JINA_EMBED_MODEL", None)
        else:
            os.environ["JINA_EMBED_MODEL"] = prev
    return [cache[text_item] for text_item in texts]


def build_client_package(client_id: str, items: list[ExperienceItem], registry: dict[str, ToolDoc], jina_client: JinaAIClient, embed_model: str) -> tuple[KnowledgePackage, str]:
    rows: list[tuple[ExperienceItem, str]] = []
    row_texts: list[str] = []
    for item in items:
        for tool in item.gold_tools:
            if tool not in registry:
                continue
            rows.append((item, tool))
            row_texts.append(item.query)
    if not rows:
        return KnowledgePackage(source_id=client_id, artifacts=[]), stable_hash([])
    row_embeddings = np.asarray(batched_query_embeddings(jina_client, row_texts, embed_model), dtype=np.float32)
    item_embeddings = np.asarray(batched_query_embeddings(jina_client, [item.query for item in items], embed_model), dtype=np.float32)
    near_misses: dict[str, list[str]] = defaultdict(list)
    row_norms = np.linalg.norm(row_embeddings, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1.0
    item_norms = np.linalg.norm(item_embeddings, axis=1, keepdims=True)
    item_norms[item_norms == 0] = 1.0
    row_embeddings = row_embeddings / row_norms
    item_embeddings = item_embeddings / item_norms
    row_query_ids = np.asarray([row_item.query_id for row_item, _tool in rows], dtype=object)
    for item_idx, item in enumerate(items):
        scores = row_embeddings @ item_embeddings[item_idx]
        scores[row_query_ids == item.query_id] = -np.inf
        predicted_tool = None
        if np.isfinite(scores).any():
            _candidate_item, predicted_tool = rows[int(np.argmax(scores))]
        if predicted_tool and predicted_tool not in item.gold_tools:
            note = f"Do not use for: {short_text(item.query, 180)}"
            if note not in near_misses[predicted_tool]:
                near_misses[predicted_tool].append(note)
    artifacts: list[KnowledgeArtifact] = []
    for item, tool in rows:
        doc = registry[tool]
        payload = {
            "type": "usage_scenario",
            "payload_mode": "typed",
            "tool_description": tool_description(doc),
            "scenario_context": short_text(item.query, 260),
            "precautions": near_misses.get(tool, [])[:3],
            "annex_summary": f"source_group={item.group}; primary_category={item.primary_category}",
        }
        metadata = {
            "tool": tool,
            "domain": tool,
            "scenario": tool,
            "category": item.primary_category,
            "source_group": item.group,
            "artifact_origin": "client_experience",
        }
        artifacts.append(
            KnowledgeArtifact(
                signature=stable_hash({"client": client_id, "tool": tool, "query_id": item.query_id}),
                text=render_usage_scenario_text(tool, payload),
                structured_payload=payload,
                metadata=metadata,
            )
        )
    package = KnowledgePackage(source_id=client_id, artifacts=artifacts, metadata={"experience_count": len(items)})
    package_hash = stable_hash([
        {"signature": artifact.signature, "tool": artifact.metadata.get("tool"), "text": artifact.text, "payload": artifact.structured_payload}
        for artifact in sorted(artifacts, key=lambda artifact: artifact.signature)
    ])
    return package, package_hash


def build_flat_pool_package(registry: dict[str, ToolDoc]) -> tuple[KnowledgePackage, str]:
    artifacts: list[KnowledgeArtifact] = []
    for tool, doc in sorted(registry.items()):
        payload = {
            "type": "tool_doc",
            "payload_mode": "typed",
            "tool_description": tool_description(doc),
            "scenario_context": short_text("; ".join(doc.descriptions[:2]), 260),
            "precautions": [],
            "annex_summary": f"doc_source=registry; categories={','.join(doc.categories[:2]) or 'unknown'}",
        }
        artifacts.append(
            KnowledgeArtifact(
                signature=stable_hash({"flat_pool": tool}),
                text=render_tool_doc_text(tool, doc),
                structured_payload=payload,
                metadata={"tool": tool, "domain": tool, "category": doc.categories[0] if doc.categories else "unknown", "artifact_origin": "tool_doc"},
            )
        )
    package = KnowledgePackage(source_id="flat_pool", artifacts=artifacts, metadata={"kind": "flat_pool"})
    return package, stable_hash([
        {"signature": artifact.signature, "tool": artifact.metadata.get("tool"), "text": artifact.text, "payload": artifact.structured_payload}
        for artifact in artifacts
    ])


def combine_packages(source_id: str, packages: list[KnowledgePackage]) -> tuple[KnowledgePackage, str]:
    artifacts: list[KnowledgeArtifact] = []
    metadata_sources: list[dict[str, Any]] = []
    for package in packages:
        artifacts.extend(package.artifacts)
        if package.metadata:
            metadata_sources.append(package.metadata)
    combined = KnowledgePackage(source_id=source_id, artifacts=artifacts, metadata={"sources": metadata_sources})
    return combined, hash_package(combined)


def serialize_package(package: KnowledgePackage) -> dict[str, Any]:
    return {
        "source_id": package.source_id,
        "metadata": package.metadata,
        "artifacts": [
            {
                "signature": artifact.signature,
                "text": artifact.text,
                "structured_payload": artifact.structured_payload,
                "metadata": artifact.metadata,
            }
            for artifact in package.artifacts
        ],
    }


def save_package(path: Path, package: KnowledgePackage, package_hash: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"package_sha256": package_hash, "package": serialize_package(package)}, indent=2), encoding="utf-8")


def deserialize_package(payload: dict[str, Any]) -> KnowledgePackage:
    package = payload["package"]
    artifacts = [
        KnowledgeArtifact(
            signature=artifact["signature"],
            text=artifact["text"],
            structured_payload=artifact.get("structured_payload"),
            metadata=artifact.get("metadata"),
        )
        for artifact in package.get("artifacts", [])
    ]
    return KnowledgePackage(source_id=package.get("source_id", "loaded"), artifacts=artifacts, metadata=package.get("metadata"))


def load_package_file(path: Path) -> tuple[KnowledgePackage, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return deserialize_package(payload), str(payload.get("package_sha256") or "")


load_saved_package = load_package_file


def read_edge_conflict_log(aggregator: EdgeAggregator) -> list[dict[str, object]]:
    value = getattr(aggregator, "conflict_log", [])
    if callable(value):
        value = value()
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    return list(value)


def trim_package_artifacts(package: KnowledgePackage, limit: int) -> KnowledgePackage:
    if limit <= 0 or len(package.artifacts) <= limit:
        return package
    return KnowledgePackage(
        source_id=package.source_id,
        artifacts=list(package.artifacts[:limit]),
        metadata=dict(package.metadata or {}),
    )


def find_seed_package_dir(roots: list[Path], seed: int) -> Path | None:
    for root in roots:
        candidate = root / "packages" / f"seed_{seed}"
        if candidate.exists():
            return candidate
        candidate = root / f"seed_{seed}"
        if candidate.exists():
            return candidate
    return None


def package_to_candidates(package: KnowledgePackage, jina_client: JinaAIClient, embed_model: str) -> list[RoutedCandidate]:
    texts = [artifact.text for artifact in package.artifacts]
    embeddings = batched_query_embeddings(jina_client, texts, embed_model) if texts else []
    candidates: list[RoutedCandidate] = []
    for artifact, embedding in zip(package.artifacts, embeddings):
        payload = artifact.structured_payload or {}
        when_to_use = []
        if isinstance(payload.get("tool_description"), str) and payload["tool_description"].strip():
            when_to_use.append(payload["tool_description"].strip())
        if isinstance(payload.get("scenario_context"), str) and payload["scenario_context"].strip():
            when_to_use.append(payload["scenario_context"].strip())
        if isinstance(payload.get("annex_summary"), str) and payload["annex_summary"].strip():
            when_to_use.append(payload["annex_summary"].strip())
        if not when_to_use:
            when_to_use = [short_text(artifact.text, 200)]
        precautions = [str(item).strip() for item in payload.get("precautions", []) if str(item).strip()] if isinstance(payload.get("precautions"), list) else []
        conflict_log = [str(item).strip() for item in payload.get("conflict_log", []) if str(item).strip()] if isinstance(payload.get("conflict_log"), list) else []
        candidates.append(
            RoutedCandidate(
                candidate_id=artifact.signature,
                label=str(artifact.metadata.get("tool") or artifact.signature),
                parent_tool=str(artifact.metadata.get("tool") or artifact.signature),
                when_to_use=when_to_use,
                do_not_use_when=render_precautions(precautions, conflict_log),
                text=artifact.text,
                embedding=embedding,
                provenance=str(artifact.metadata.get("artifact_origin") or artifact.metadata.get("source_group") or artifact.metadata.get("category") or "stabletoolbench"),
            )
        )
    return candidates


def hash_package(package: KnowledgePackage) -> str:
    return stable_hash([
        {"signature": artifact.signature, "tool": artifact.metadata.get("tool"), "text": artifact.text, "structured_payload": artifact.structured_payload}
        for artifact in sorted(package.artifacts, key=lambda artifact: artifact.signature)
    ])


def train_query_classifier(items: list[ExperienceItem]) -> ClassifierBundle | None:
    if TfidfVectorizer is None or OneVsRestClassifier is None or MultiLabelBinarizer is None or LinearSVC is None:
        return None
    texts = [item.query for item in items]
    labels = [item.gold_tools for item in items]
    if not texts:
        return None
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(labels)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X = vectorizer.fit_transform(texts)
    clf = OneVsRestClassifier(LinearSVC())
    clf.fit(X, y)
    return ClassifierBundle(vectorizer=vectorizer, classifier=clf, classes=list(mlb.classes_))


def predict_query_classifier(bundle: ClassifierBundle | None, query: str) -> str:
    if bundle is None:
        return ""
    scores = bundle.classifier.decision_function(bundle.vectorizer.transform([query]))
    values = scores.tolist()[0] if hasattr(scores, "tolist") and isinstance(scores.tolist()[0], list) else scores.tolist()
    best_index = max(range(len(values)), key=lambda idx: values[idx])
    return bundle.classes[best_index]


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def maybe_cuda_synchronize(backend: Any) -> None:
    try:
        import torch
        device = getattr(getattr(backend, "model", None), "device", None)
        if device is not None and getattr(device, "type", None) == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)
    except Exception:
        return


def _select_distinct_tools(ranked_indices: list[int], candidates: list[RoutedCandidate], limit: int) -> list[str]:
    selected_tools: list[str] = []
    seen_tools: set[str] = set()
    for idx in ranked_indices:
        tool_name = candidates[idx].parent_tool
        if tool_name in seen_tools:
            continue
        seen_tools.add(tool_name)
        selected_tools.append(tool_name)
        if len(selected_tools) >= limit:
            break
    return selected_tools


def _expand_ranked_candidates(ranked_indices: list[int], candidates: list[RoutedCandidate], selected_tools: list[str]) -> list[RoutedCandidate]:
    expanded: list[RoutedCandidate] = []
    counts = {tool_name: 0 for tool_name in selected_tools}
    for idx in ranked_indices:
        candidate = candidates[idx]
        if candidate.parent_tool not in counts:
            continue
        if counts[candidate.parent_tool] >= 2:
            continue
        expanded.append(candidate)
        counts[candidate.parent_tool] += 1
    return expanded


def build_ranked_candidates(candidates: list[RoutedCandidate], similarities: np.ndarray, retrieval_pool_size: int, top_k: int, retrieval_mode: str) -> tuple[list[RoutedCandidate], list[str]]:
    pool_size = max(top_k, retrieval_pool_size)
    ranked_indices = np.argsort(-similarities).tolist()
    ranked_pool = ranked_indices[:pool_size]
    pool_tools = [candidates[idx].parent_tool for idx in ranked_pool]
    if retrieval_mode == "distinct_tool_topk":
        selected_tools = _select_distinct_tools(ranked_pool, candidates, top_k)
        return _expand_ranked_candidates(ranked_pool, candidates, selected_tools), pool_tools
    if retrieval_mode == "union_doc_scenario":
        doc_indices = [idx for idx in ranked_indices if candidates[idx].provenance == "tool_doc"][:pool_size]
        scenario_indices = [idx for idx in ranked_indices if candidates[idx].provenance != "tool_doc"][:pool_size]
        doc_tools = _select_distinct_tools(doc_indices, candidates, top_k)
        scenario_tools = _select_distinct_tools(scenario_indices, candidates, top_k)
        union_tools = list(dict.fromkeys(doc_tools + scenario_tools))
        best_score_by_tool: dict[str, float] = {}
        for idx in ranked_indices:
            tool_name = candidates[idx].parent_tool
            if tool_name not in union_tools:
                continue
            if tool_name not in best_score_by_tool:
                best_score_by_tool[tool_name] = float(similarities[idx])
        selected_tools = sorted(union_tools, key=lambda tool_name: best_score_by_tool.get(tool_name, float("-inf")), reverse=True)[:top_k]
        return _expand_ranked_candidates(ranked_indices, candidates, selected_tools), selected_tools
    return [candidates[idx] for idx in ranked_pool[:top_k]], pool_tools


def candidate_selection_diagnostics(
    candidates: list[RoutedCandidate],
    similarities: np.ndarray,
    ranked: list[RoutedCandidate],
    top_k: int,
    retrieval_pool_size: int,
    retrieval_mode: str,
) -> dict[str, Any]:
    candidate_tools = [candidate.parent_tool for candidate in ranked]
    distinct_tools = list(dict.fromkeys(candidate_tools))
    selected_tools = set(distinct_tools)
    depth_reached = 0
    if len(similarities):
        for depth_reached, candidate_idx in enumerate(np.argsort(-similarities).tolist(), start=1):
            selected_tools.discard(candidates[candidate_idx].parent_tool)
            if not selected_tools:
                break
    return {
        "candidate_rule": "distinct5_walkdown" if retrieval_mode == "distinct_tool_topk" else retrieval_mode,
        "candidate_distinct_count": len(distinct_tools),
        "candidate_depth_reached": depth_reached,
        "candidate_shortfall": len(distinct_tools) < top_k,
        "candidate_top_k": top_k,
        "candidate_retrieval_cap": retrieval_pool_size,
    }


def summarize_latency(rows: list[dict[str, Any]]) -> dict[str, float]:
    retrieval = [float(row.get("retrieval_s", 0.0)) for row in rows]
    rerank = [float(row.get("rerank_s", 0.0)) for row in rows]
    total = [float(row.get("total_s", 0.0)) for row in rows]
    return {
        "retrieval_p50_s": percentile(retrieval, 0.5),
        "retrieval_p95_s": percentile(retrieval, 0.95),
        "rerank_p50_s": percentile(rerank, 0.5),
        "rerank_p95_s": percentile(rerank, 0.95),
        "total_p50_s": percentile(total, 0.5),
        "total_p95_s": percentile(total, 0.95),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["group"]].append(row)
    group_metrics = {}
    for group, bucket in sorted(grouped.items()):
        group_metrics[group] = {
            "count": len(bucket),
            "accuracy": sum(1 for row in bucket if row["routed_correctly"]) / len(bucket),
            "recall_at_5": sum(1 for row in bucket if row["gold_in_top_k"]) / len(bucket),
        }
        group_metrics[group].update(summarize_latency(bucket))
    summary = {
        "query_count": len(rows),
        "accuracy": sum(1 for row in rows if row["routed_correctly"]) / len(rows) if rows else 0.0,
        "recall_at_5": sum(1 for row in rows if row["gold_in_top_k"]) / len(rows) if rows else 0.0,
        "group_metrics": group_metrics,
    }
    summary.update(summarize_latency(rows))
    return summary


def evaluate_reranker_arm(
    name: str,
    package: KnowledgePackage,
    test_items: list[TestQuery],
    jina_client: JinaAIClient,
    embed_model: str,
    backend: Any,
    top_k: int,
    retrieval_pool_size: int,
    retrieval_mode: str,
    reranker_variant: str,
    *,
    progress: ProgressLogger | None = None,
    seed: int | None = None,
    heartbeat_every: int = 25,
) -> dict[str, Any]:
    if progress is not None:
        progress.log(
            "arm_prepare_begin",
            seed=seed,
            arm=name,
            artifact_count=len(package.artifacts),
            query_count=len(test_items),
        )
    prepare_started = time.perf_counter()
    candidates = package_to_candidates(package, jina_client, embed_model)
    candidate_prepare_s = time.perf_counter() - prepare_started
    if progress is not None:
        progress.log(
            "arm_candidates_ready",
            seed=seed,
            arm=name,
            artifact_count=len(package.artifacts),
            candidate_count=len(candidates),
            elapsed_prepare_s=candidate_prepare_s,
        )
    query_embed_started = time.perf_counter()
    query_embeddings = batched_query_embeddings(jina_client, [item.query for item in test_items], embed_model) if test_items else []
    query_embed_s = time.perf_counter() - query_embed_started
    if progress is not None:
        progress.log(
            "arm_queries_embedded",
            seed=seed,
            arm=name,
            query_count=len(query_embeddings),
            elapsed_query_embed_s=query_embed_s,
            total_prepare_s=candidate_prepare_s + query_embed_s,
        )
    candidate_matrix = np.asarray([candidate.embedding for candidate in candidates], dtype=np.float32) if candidates else np.zeros((0, 0), dtype=np.float32)
    if candidate_matrix.size:
        candidate_norms = np.linalg.norm(candidate_matrix, axis=1)
        candidate_norms[candidate_norms == 0.0] = 1.0
        candidate_matrix = candidate_matrix / candidate_norms[:, None]
    rows: list[dict[str, Any]] = []
    total_latency = 0.0
    total_retrieval = 0.0
    total_rerank = 0.0
    parse_failures = 0
    for idx, (item, embedding) in enumerate(zip(test_items, query_embeddings), start=1):
        total_started = time.perf_counter()
        retrieval_started = time.perf_counter()
        query_vector = np.asarray(embedding, dtype=np.float32)
        query_norm = float(np.linalg.norm(query_vector))
        if query_norm > 0.0:
            query_vector = query_vector / query_norm
        similarities = candidate_matrix @ query_vector if candidate_matrix.size else np.asarray([], dtype=np.float32)
        ranked, retrieval_pool_tools = build_ranked_candidates(candidates, similarities, retrieval_pool_size, top_k, retrieval_mode)
        candidate_diag = candidate_selection_diagnostics(candidates, similarities, ranked, top_k, retrieval_pool_size, retrieval_mode)
        if candidate_diag["candidate_shortfall"] and retrieval_mode == "distinct_tool_topk":
            raise RuntimeError(
                f"{name} seed {seed} query {item.query_id} returned "
                f"{candidate_diag['candidate_distinct_count']} distinct candidates under cap {retrieval_pool_size}"
            )
        retrieval_latency = time.perf_counter() - retrieval_started
        top_candidate = ranked[0] if ranked else None
        if not ranked or top_candidate is None:
            total_latency_value = time.perf_counter() - total_started
            rows.append({"query_id": item.query_id, "query_text": item.query, "group": item.group, "gold_parent_tools": item.gold_tools, "predicted_tool": "", "routed_correctly": False, "gold_in_top_k": False, "top_candidates": [], "top_candidate_ids": [], "retrieval_pool_tools": retrieval_pool_tools, **candidate_diag, "parse_ok": False, "fallback_used": True, "latency_seconds": total_latency_value, "retrieval_s": retrieval_latency, "rerank_s": 0.0, "total_s": total_latency_value})
            total_latency += total_latency_value
            total_retrieval += retrieval_latency
            continue
        maybe_cuda_synchronize(backend)
        rerank_started = time.perf_counter()
        result = run_prompt(backend, reranker_variant, "toolbench", item.query, ranked, [], top_candidate)
        maybe_cuda_synchronize(backend)
        rerank_latency = time.perf_counter() - rerank_started
        total_latency_value = time.perf_counter() - total_started
        total_latency += total_latency_value
        total_retrieval += retrieval_latency
        total_rerank += rerank_latency
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
            "gold_in_top_k": any(tool in item.gold_tools for tool in candidate_tools),
            "top_candidates": candidate_tools,
            "top_candidate_ids": [candidate.candidate_id for candidate in ranked],
            "retrieval_pool_tools": retrieval_pool_tools,
            **candidate_diag,
            "parse_ok": result.parse_ok,
            "fallback_used": result.fallback_used,
            "latency_seconds": total_latency_value,
            "retrieval_s": retrieval_latency,
            "rerank_s": rerank_latency,
            "total_s": total_latency_value,
            "prompt_hash": result.prompt_hash,
        })
        if progress is not None and (idx == 1 or idx % heartbeat_every == 0 or idx == len(test_items)):
            progress.log(
                "arm_progress",
                seed=seed,
                arm=name,
                completed_queries=idx,
                total_queries=len(test_items),
                latest_query_id=item.query_id,
                running_accuracy=(sum(1 for row in rows if row["routed_correctly"]) / len(rows)) if rows else 0.0,
                running_recall_at_5=(sum(1 for row in rows if row["gold_in_top_k"]) / len(rows)) if rows else 0.0,
                mean_total_s=(total_latency / len(rows)) if rows else 0.0,
                mean_retrieval_s=(total_retrieval / len(rows)) if rows else 0.0,
                mean_rerank_s=(total_rerank / len(rows)) if rows else 0.0,
            )
    summary = summarize_rows(rows)
    summary.update({
        "arm": name,
        "mean_latency_seconds": total_latency / len(test_items) if test_items else 0.0,
        "mean_retrieval_seconds": total_retrieval / len(test_items) if test_items else 0.0,
        "mean_rerank_seconds": total_rerank / len(test_items) if test_items else 0.0,
        "parse_failure_rate": parse_failures / len(test_items) if test_items else 0.0,
        "retrieval_mode": retrieval_mode,
        "retrieval_pool_size": retrieval_pool_size,
        "reranker_variant": reranker_variant,
        "candidate_count": len(candidates),
        "candidate_prepare_s": candidate_prepare_s,
        "query_prepare_s": query_embed_s,
        "rows": rows,
    })
    return summary


def evaluate_classifier_arm(bundle: ClassifierBundle | None, test_items: list[TestQuery]) -> dict[str, Any]:
    rows = []
    for item in test_items:
        pred = predict_query_classifier(bundle, item.query)
        rows.append({"query_id": item.query_id, "query_text": item.query, "group": item.group, "gold_parent_tools": item.gold_tools, "predicted_tool": pred, "routed_correctly": pred in item.gold_tools, "gold_in_top_k": False, "top_candidates": [], "parse_ok": True, "fallback_used": False, "latency_seconds": 0.0, "retrieval_s": 0.0, "rerank_s": 0.0, "total_s": 0.0})
    summary = summarize_rows(rows)
    summary.update({"arm": "query_classifier", "mean_latency_seconds": 0.0, "mean_retrieval_seconds": 0.0, "mean_rerank_seconds": 0.0, "parse_failure_rate": 0.0, "rows": rows})
    return summary


def evaluate_local_only_arm(client_packages: dict[str, KnowledgePackage], tool_doc_package: KnowledgePackage, test_items: list[TestQuery], jina_client: JinaAIClient, embed_model: str, backend: Any, top_k: int, retrieval_pool_size: int, retrieval_mode: str, reranker_variant: str, *, progress: ProgressLogger | None = None, seed: int | None = None) -> dict[str, Any]:
    client_results: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for client_id, package in sorted(client_packages.items()):
        local_package, local_hash = combine_packages(f"{client_id}_with_docs", [tool_doc_package, package])
        result = evaluate_reranker_arm("local_only", local_package, test_items, jina_client, embed_model, backend, top_k, retrieval_pool_size, retrieval_mode, reranker_variant, progress=progress, seed=seed)
        result.update({"client_id": client_id, "compendium_sha256": local_hash, "artifact_count": len(local_package.artifacts)})
        client_results.append(result)
        all_rows.extend(result["rows"])
    combined_rows = summarize_rows(all_rows)
    return {
        "arm": "local_only",
        "accuracy": statistics.mean(result["accuracy"] for result in client_results) if client_results else 0.0,
        "recall_at_5": statistics.mean(result["recall_at_5"] for result in client_results) if client_results else 0.0,
        "mean_latency_seconds": statistics.mean(result["mean_latency_seconds"] for result in client_results) if client_results else 0.0,
        "mean_retrieval_seconds": statistics.mean(result.get("mean_retrieval_seconds", 0.0) for result in client_results) if client_results else 0.0,
        "mean_rerank_seconds": statistics.mean(result.get("mean_rerank_seconds", 0.0) for result in client_results) if client_results else 0.0,
        "parse_failure_rate": statistics.mean(result["parse_failure_rate"] for result in client_results) if client_results else 0.0,
        "retrieval_mode": retrieval_mode,
        "retrieval_pool_size": retrieval_pool_size,
        "reranker_variant": reranker_variant,
        "group_metrics": combined_rows["group_metrics"],
        "retrieval_p50_s": combined_rows["retrieval_p50_s"],
        "retrieval_p95_s": combined_rows["retrieval_p95_s"],
        "rerank_p50_s": combined_rows["rerank_p50_s"],
        "rerank_p95_s": combined_rows["rerank_p95_s"],
        "total_p50_s": combined_rows["total_p50_s"],
        "total_p95_s": combined_rows["total_p95_s"],
        "rows": all_rows,
        "client_results": client_results,
    }


def aggregate_seed_summaries(seed_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    accuracies = [summary["accuracy"] for summary in seed_summaries]
    recalls = [summary["recall_at_5"] for summary in seed_summaries]
    groups = sorted({group for summary in seed_summaries for group in summary["group_metrics"]})
    group_metrics = {}
    per_group_metrics = {}
    for group in groups:
        accs = [summary["group_metrics"][group]["accuracy"] for summary in seed_summaries if group in summary["group_metrics"]]
        recs = [summary["group_metrics"][group]["recall_at_5"] for summary in seed_summaries if group in summary["group_metrics"]]
        counts = [summary["group_metrics"][group]["count"] for summary in seed_summaries if group in summary["group_metrics"]]
        group_metrics[group] = {
            "mean_accuracy": statistics.mean(accs) if accs else 0.0,
            "sd_accuracy": statistics.stdev(accs) if len(accs) > 1 else 0.0,
            "mean_recall_at_5": statistics.mean(recs) if recs else 0.0,
            "count": counts[0] if counts else 0,
        }
        per_group_metrics[group] = {
            "n": counts[0] if counts else 0,
            "accuracy": {
                "mean": statistics.mean(accs) if accs else 0.0,
                "sd": statistics.stdev(accs) if len(accs) > 1 else 0.0,
            },
            "recall_at_5": {
                "mean": statistics.mean(recs) if recs else 0.0,
                "sd": statistics.stdev(recs) if len(recs) > 1 else 0.0,
            },
        }
    return {
        "mean_accuracy": statistics.mean(accuracies) if accuracies else 0.0,
        "sd_accuracy": statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
        "mean_recall_at_5": statistics.mean(recalls) if recalls else 0.0,
        "sd_recall_at_5": statistics.stdev(recalls) if len(recalls) > 1 else 0.0,
        "group_metrics": group_metrics,
        "metrics": {
            "accuracy": {
                "mean": statistics.mean(accuracies) if accuracies else 0.0,
                "sd": statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
            },
            "recall_at_5": {
                "mean": statistics.mean(recalls) if recalls else 0.0,
                "sd": statistics.stdev(recalls) if len(recalls) > 1 else 0.0,
            },
            "per_group": per_group_metrics,
        },
    }


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    args = parse_args()
    commit, dirty_entries = assert_clean_tree(allow_dirty=args.allow_dirty)
    groups = parse_csv(args.groups)
    arms = parse_csv(args.arms)
    seeds = parse_seed_list(args.seeds)
    load_package_roots = [Path(part) for part in parse_csv(args.load_packages_roots)]
    if args.self_test and not load_package_roots:
        raise RuntimeError("--self-test requires --load-packages-roots so the persisted package loading path is exercised.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress = ProgressLogger(args.output_dir)
    progress.log(
        "start",
        groups=groups,
        arms=arms,
        seeds=seeds,
        data_mode=args.data_mode,
        partition_mode=args.partition_mode,
        client_count=args.client_count,
        top_k=args.top_k,
        retrieval_mode=args.retrieval_mode,
        retrieval_pool_size=args.retrieval_pool_size,
        reranker_variant=args.reranker_variant,
        junk_filter=args.junk_filter,
        build_only=args.build_only,
        load_packages_roots=[str(path) for path in load_package_roots],
        skip_package_persistence=args.skip_package_persistence,
        git_commit=commit,
        allow_dirty=args.allow_dirty,
        dirty_entry_count=len(dirty_entries),
        dirty_entries_sample=dirty_entries[:20],
    )

    progress.log("load_queries_begin", stb_root=str(args.stb_root))
    queries = load_stabletoolbench_queries(args.stb_root, groups)
    progress.log("load_queries_done", query_count=len(queries))
    if args.data_mode == "toolbench_train":
        progress.log("load_training_begin", instruction_dir=str(args.toolbench_instruction_dir))
        train_items = load_toolbench_training_items(args.toolbench_instruction_dir)
        base_registry = build_tool_registry(queries + train_items)
        progress.log("load_training_done", train_count=len(train_items), registry_tool_count=len(base_registry))
    else:
        train_items = []
        base_registry = build_tool_registry(queries)
        progress.log("build_registry_done", registry_tool_count=len(base_registry))

    junk_filter_info = {"enabled": False, "dropped_tool_count": 0, "dropped_tools_sample": [], "dropped_query_count": 0, "pruned_gold_tool_refs": 0, "dropped_train_item_count": 0, "pruned_train_gold_tool_refs": 0}
    if args.junk_filter:
        base_registry, registry_filter_info = apply_junk_filter(base_registry)
        queries, query_filter_info = filter_eval_queries(queries, base_registry)
        train_items, train_filter_info = filter_experience_items(train_items, base_registry)
        junk_filter_info = {**registry_filter_info, **query_filter_info, **train_filter_info}
        progress.log("junk_filter_done", registry_tool_count=len(base_registry), query_count=len(queries), train_count=len(train_items), **junk_filter_info)

    contamination_filter_info = {
        "enabled": args.data_mode == "toolbench_train",
        "pool_items_removed_exact": 0,
        "pool_items_removed_near_dup": 0,
        "pool_items_removed_total": 0,
        "eval_queries_with_exact_match_in_pool": 0,
        "eval_queries_with_near_dup_in_pool": 0,
        "eval_queries_removed": 0,
        "near_duplicate_threshold": args.contamination_near_duplicate_threshold,
        "report_threshold": args.contamination_report_threshold,
        "near_duplicate_removed_count_at_report_threshold": 0,
        "near_duplicate_removed_count_if_threshold_0_90": 0,
        "post_filter_refusal_check": None,
        "pool_items_removed_exact_query_ids_sample": [],
        "eval_queries_with_exact_match_in_pool_ids_sample": [],
        "pool_items_removed_near_dup_query_ids_sample": [],
        "eval_queries_with_near_dup_in_pool_ids_sample": [],
        "near_duplicate_removed_pairs_sample": [],
    }

    if args.data_mode == "toolbench_train" and args.max_train_items > 0:
        original_train_count = len(train_items)
        smoke_seed = seeds[0] if seeds else 0
        train_items = limit_experience_items(train_items, args.max_train_items, smoke_seed)
        contamination_filter_info["pre_filter_train_cap"] = {
            "enabled": True,
            "original_train_count": original_train_count,
            "capped_train_count": len(train_items),
            "max_train_items": args.max_train_items,
            "seed": smoke_seed,
        }
        progress.log(
            "pre_filter_train_cap_applied",
            original_train_count=original_train_count,
            capped_train_count=len(train_items),
            max_train_items=args.max_train_items,
            seed=smoke_seed,
        )
    else:
        contamination_filter_info["pre_filter_train_cap"] = {"enabled": False}

    embedder_info = resolve_local_embedder()
    jina_keys = os.environ.get("JINA_API_KEY") and [os.environ["JINA_API_KEY"]] or []
    jina_client = JinaAIClient(api_keys=jina_keys)
    progress.log(
        "init_embedding_client_done",
        embed_model=args.embed_model,
        local_model=embedder_info["model_path"],
        embedder_revision=embedder_info["revision"],
        embedder_device=embedder_info["device"],
    )

    with temporary_env({
        "JINA_LOCAL_EMBED_MODEL": embedder_info["model_path"],
        "JINA_LOCAL_EMBED_DEVICE": embedder_info["device"],
        "JINA_LOCAL_EMBED_LOCAL_ONLY": embedder_info["local_only"],
        "JINA_API_KEY": None,
    }):
        forbidden_near_duplicate_texts: set[str] = set()
        if args.data_mode == "toolbench_train":
            train_items, exact_overlap_info = remove_exact_eval_overlaps(train_items, queries)
            train_items, near_overlap_info, forbidden_near_duplicate_texts = remove_near_duplicate_eval_overlaps(
                train_items,
                queries,
                jina_client,
                args.embed_model,
                removal_threshold=args.contamination_near_duplicate_threshold,
                report_threshold=args.contamination_report_threshold,
                pool_embedding_cache_dir=args.pool_embedding_cache_dir,
                progress=progress,
            )
            near_overlap_info.pop("pool_items_removed_near_dup_query_ids", None)
            contamination_filter_info = {**contamination_filter_info, **exact_overlap_info, **near_overlap_info}
            contamination_filter_info["pool_items_removed_total"] = (
                int(contamination_filter_info["pool_items_removed_exact"])
                + int(contamination_filter_info["pool_items_removed_near_dup"])
            )
            contamination_filter_info["eval_queries_removed"] = 0
            refusal_info = assert_no_eval_overlap(train_items, queries, forbidden_near_duplicate_texts=forbidden_near_duplicate_texts, stage="post_filter_training_pool")
            contamination_filter_info["post_filter_refusal_check"] = refusal_info
            progress.log("contamination_filter_done", train_count=len(train_items), test_count=len(queries), **contamination_filter_info)
        else:
            progress.log("contamination_filter_skipped", enabled=False, data_mode=args.data_mode)

        backend = None
        reranker_arms = {"synapse", "centralized", "local_only", "flat_pool"}
        if not args.build_only and any(arm in reranker_arms for arm in arms):
            progress.log("load_backend_begin", model_path=args.model_path)
            backend = load_local_backend(args.model_path)
            progress.log("load_backend_done")

        results_by_arm = {arm: [] for arm in arms}
        synapse_hashes: list[str] = []

        drift_seed = seeds[0] + args.doc_drift_seed_offset if seeds else args.doc_drift_seed_offset
        registry, doc_drift_info = apply_doc_drift(
            base_registry,
            fraction=args.doc_drift_fraction,
            mode=args.doc_drift_mode,
            seed=drift_seed,
        )
        progress.log("doc_drift_ready", **doc_drift_info)

        for seed in seeds:
            progress.log("seed_begin", seed=seed)
            if args.data_mode == "toolbench_train":
                seed_train_items = list(train_items)
                test_items = queries
            else:
                seed_train_items, test_items = split_experience_test(queries, args.test_fraction, seed)
            if args.max_train_items > 0:
                original_train_count = len(seed_train_items)
                seed_train_items = limit_experience_items(seed_train_items, args.max_train_items, seed)
                progress.log("train_cap_applied", seed=seed, original_train_count=original_train_count, capped_train_count=len(seed_train_items), max_train_items=args.max_train_items)
            progress.log("seed_split_done", seed=seed, train_count=len(seed_train_items), test_count=len(test_items))
            seed_train_refusal_info = None
            classifier_fit_refusal_info = None
            if args.data_mode == "toolbench_train":
                seed_train_refusal_info = assert_no_eval_overlap(seed_train_items, test_items, forbidden_near_duplicate_texts=forbidden_near_duplicate_texts, stage=f"seed_{seed}_training_pool")
                classifier_fit_refusal_info = assert_no_eval_overlap(seed_train_items, test_items, forbidden_near_duplicate_texts=forbidden_near_duplicate_texts, stage=f"seed_{seed}_classifier_fit_set")
                progress.log("seed_contamination_refusal_passed", seed=seed, training_pool=seed_train_refusal_info, classifier_fit_set=classifier_fit_refusal_info)
            clients = assign_clients(seed_train_items, args.client_count, args.partition_mode, seed)
            if args.max_items_per_client > 0:
                original_client_sizes = {client_id: len(items) for client_id, items in sorted(clients.items())}
                clients = limit_client_items(clients, args.max_items_per_client, seed)
                progress.log("client_cap_applied", seed=seed, original_client_sizes=original_client_sizes, capped_client_sizes={client_id: len(items) for client_id, items in sorted(clients.items())}, max_items_per_client=args.max_items_per_client)
            progress.log("client_partition_done", seed=seed, client_sizes={client_id: len(items) for client_id, items in sorted(clients.items())})
            seed_package_dir = find_seed_package_dir(load_package_roots, seed) if load_package_roots else None
            client_packages: dict[str, KnowledgePackage] = {}
            client_hashes: list[str] = []
            tool_doc_package: KnowledgePackage
            tool_doc_hash: str
            flat_package: KnowledgePackage
            flat_hash: str
            if seed_package_dir is not None:
                progress.log("seed_package_load_begin", seed=seed, packages_dir=str(seed_package_dir))
                for client_id, items in sorted(clients.items()):
                    package_path = seed_package_dir / f"{client_id}.json"
                    if not package_path.exists():
                        raise FileNotFoundError(f"Missing persisted client package for seed {seed}: {package_path}")
                    package, package_hash = load_saved_package(package_path)
                    client_packages[client_id] = package
                    client_hashes.append(package_hash)
                    progress.log("client_package_loaded", seed=seed, client_id=client_id, item_count=len(items), artifact_count=len(package.artifacts), package_sha256=package_hash)
                tool_doc_path = seed_package_dir / "tool_docs.json"
                if not tool_doc_path.exists():
                    raise FileNotFoundError(f"Missing persisted tool doc package for seed {seed}: {tool_doc_path}")
                tool_doc_package, tool_doc_hash = load_saved_package(tool_doc_path)
                progress.log("tool_doc_package_loaded", seed=seed, artifact_count=len(tool_doc_package.artifacts), package_sha256=tool_doc_hash)
                flat_path = seed_package_dir / "flat_pool.json"
                if not flat_path.exists():
                    raise FileNotFoundError(f"Missing persisted flat pool package for seed {seed}: {flat_path}")
                flat_package, flat_hash = load_saved_package(flat_path)
                progress.log("flat_pool_package_loaded", seed=seed, artifact_count=len(flat_package.artifacts), package_sha256=flat_hash)
            else:
                for client_id, items in sorted(clients.items()):
                    progress.log("client_package_begin", seed=seed, client_id=client_id, item_count=len(items))
                    package, package_hash = build_client_package(client_id, items, registry, jina_client, args.embed_model)
                    client_packages[client_id] = package
                    client_hashes.append(package_hash)
                    if not args.skip_package_persistence:
                        save_package(args.output_dir / "packages" / f"seed_{seed}" / f"{client_id}.json", package, package_hash)
                    progress.log("client_package_done", seed=seed, client_id=client_id, artifact_count=len(package.artifacts), package_sha256=package_hash)
                progress.log("tool_doc_package_begin", seed=seed, registry_tool_count=len(registry))
                tool_doc_package, tool_doc_hash = build_doc_package(registry)
                if not args.skip_package_persistence:
                    save_package(args.output_dir / "packages" / f"seed_{seed}" / "tool_docs.json", tool_doc_package, tool_doc_hash)
                progress.log("tool_doc_package_done", seed=seed, artifact_count=len(tool_doc_package.artifacts), package_sha256=tool_doc_hash)
                flat_package, flat_hash = build_flat_pool_package(registry)
                if not args.skip_package_persistence:
                    save_package(args.output_dir / "packages" / f"seed_{seed}" / "flat_pool.json", flat_package, flat_hash)
                progress.log("flat_pool_package_done", seed=seed, artifact_count=len(flat_package.artifacts), package_sha256=flat_hash)
                if args.build_only:
                    progress.log("seed_build_complete", seed=seed, client_package_count=len(client_packages), registry_tool_count=len(registry))
                    continue
            classifier_bundle = train_query_classifier(seed_train_items) if "query_classifier" in arms else None
            if "query_classifier" in arms:
                progress.log("query_classifier_ready", seed=seed, enabled=classifier_bundle is not None)

            if args.self_test:
                original_query_count = len(test_items)
                test_items = test_items[: min(5, len(test_items))]
                client_packages = {client_id: trim_package_artifacts(package, 128) for client_id, package in client_packages.items()}
                tool_doc_package = trim_package_artifacts(tool_doc_package, 256)
                flat_package = trim_package_artifacts(flat_package, 256)
                progress.log(
                    "self_test_slice",
                    seed=seed,
                    original_query_count=original_query_count,
                    self_test_query_count=len(test_items),
                    self_test_client_artifact_count=sum(len(package.artifacts) for package in client_packages.values()),
                    self_test_tool_doc_artifact_count=len(tool_doc_package.artifacts),
                    self_test_flat_artifact_count=len(flat_package.artifacts),
                    arms=arms,
                )

            common_run_metadata = {
                "seed": seed,
                "data_mode": args.data_mode,
                "paper_eligible": args.data_mode == "toolbench_train",
                "train_count": len(seed_train_items),
                "test_count": len(test_items),
                "metric_definition": {"correct": "predicted_tool in gold_parent_tools", "recall_at_5": "any gold tool present among candidate tools"},
                "doc_drift": doc_drift_info,
                "junk_filter": junk_filter_info,
                "contamination_filter": contamination_filter_info,
                "seed_train_refusal_check": seed_train_refusal_info,
                "classifier_fit_refusal_check": classifier_fit_refusal_info,
            }

            if "synapse" in arms:
                progress.log("arm_begin", seed=seed, arm="synapse", merge_policy=args.merge_policy)
                progress.log(
                    "arm_merge_begin",
                    seed=seed,
                    arm="synapse",
                    merge_policy=args.merge_policy,
                    client_package_count=len(client_packages),
                    client_artifact_count=sum(len(package.artifacts) for package in client_packages.values()),
                )
                merge_started = time.perf_counter()
                with temporary_env({"SYNAPSE_EDGE_MERGE_POLICY": args.merge_policy}):
                    aggregator = EdgeAggregator(EdgeConfig(edge_id=f"stabletoolbench_seed_{seed}"))
                    with merge_heartbeat(progress, seed=seed, arm="synapse", merge_policy=args.merge_policy):
                        merged = aggregator.merge_packages(list(client_packages.values()))
                merge_elapsed = time.perf_counter() - merge_started
                if merged is None:
                    raise RuntimeError(f"Seed {seed} synapse merge produced no package")
                progress.log(
                    "arm_merge_done",
                    seed=seed,
                    arm="synapse",
                    merge_policy=args.merge_policy,
                    elapsed_merge_s=merge_elapsed,
                    merged_artifact_count=len(merged.artifacts),
                )
                synapse_package, synapse_hash = combine_packages("synapse_with_docs", [tool_doc_package, merged])
                synapse_hashes.append(synapse_hash)
                synapse_result = evaluate_reranker_arm("synapse", synapse_package, test_items, jina_client, args.embed_model, backend, args.top_k, args.retrieval_pool_size, args.retrieval_mode, args.reranker_variant, progress=progress, seed=seed)
                synapse_result.update(common_run_metadata)
                synapse_result.update({
                    "compendium": {
                        "global_sha256": synapse_hash,
                        "client_sha256": client_hashes,
                        "tool_doc_sha256": tool_doc_hash,
                        "flat_pool_sha256": flat_hash,
                        "artifact_count": len(synapse_package.artifacts),
                    },
                    "embedder": embedder_info,
                    "merge_policy": args.merge_policy,
                    "edge_conflict_log": read_edge_conflict_log(aggregator),
                })
                results_by_arm["synapse"].append(synapse_result)
                save_json(args.output_dir / f"seed_{seed}" / "synapse.json", synapse_result)
                progress.log("arm_done", seed=seed, arm="synapse", accuracy=synapse_result["accuracy"], recall_at_5=synapse_result["recall_at_5"])

            if "centralized" in arms:
                progress.log("arm_begin", seed=seed, arm="centralized")
                pooled_experience, pooled_hash = combine_packages("centralized_experience", list(client_packages.values()))
                centralized_package, centralized_hash = combine_packages("centralized_with_docs", [tool_doc_package, pooled_experience])
                centralized_result = evaluate_reranker_arm("centralized", centralized_package, test_items, jina_client, args.embed_model, backend, args.top_k, args.retrieval_pool_size, args.retrieval_mode, args.reranker_variant, progress=progress, seed=seed)
                centralized_result.update(common_run_metadata)
                centralized_result.update({
                    "compendium": {
                        "global_sha256": centralized_hash,
                        "pooled_experience_sha256": pooled_hash,
                        "client_sha256": client_hashes,
                        "tool_doc_sha256": tool_doc_hash,
                        "flat_pool_sha256": flat_hash,
                        "artifact_count": len(centralized_package.artifacts),
                    },
                    "embedder": embedder_info,
                })
                results_by_arm["centralized"].append(centralized_result)
                save_json(args.output_dir / f"seed_{seed}" / "centralized.json", centralized_result)
                progress.log("arm_done", seed=seed, arm="centralized", accuracy=centralized_result["accuracy"], recall_at_5=centralized_result["recall_at_5"])

            if "local_only" in arms:
                progress.log("arm_begin", seed=seed, arm="local_only")
                local_result = evaluate_local_only_arm(client_packages, tool_doc_package, test_items, jina_client, args.embed_model, backend, args.top_k, args.retrieval_pool_size, args.retrieval_mode, args.reranker_variant, progress=progress, seed=seed)
                local_result.update(common_run_metadata)
                local_result.update({
                    "compendium": {
                        "client_sha256": client_hashes,
                        "tool_doc_sha256": tool_doc_hash,
                    },
                    "embedder": embedder_info,
                })
                results_by_arm["local_only"].append(local_result)
                save_json(args.output_dir / f"seed_{seed}" / "local_only.json", local_result)
                progress.log("arm_done", seed=seed, arm="local_only", accuracy=local_result["accuracy"], recall_at_5=local_result["recall_at_5"])

            if "flat_pool" in arms:
                progress.log("arm_begin", seed=seed, arm="flat_pool")
                flat_result = evaluate_reranker_arm("flat_pool", flat_package, test_items, jina_client, args.embed_model, backend, args.top_k, args.retrieval_pool_size, args.retrieval_mode, args.reranker_variant, progress=progress, seed=seed)
                flat_result.update(common_run_metadata)
                flat_result.update({
                    "compendium": {
                        "global_sha256": flat_hash,
                        "artifact_count": len(flat_package.artifacts),
                    },
                    "embedder": embedder_info,
                })
                results_by_arm["flat_pool"].append(flat_result)
                save_json(args.output_dir / f"seed_{seed}" / "flat_pool.json", flat_result)
                progress.log("arm_done", seed=seed, arm="flat_pool", accuracy=flat_result["accuracy"], recall_at_5=flat_result["recall_at_5"])

            if "query_classifier" in arms:
                progress.log("arm_begin", seed=seed, arm="query_classifier")
                classifier_result = evaluate_classifier_arm(classifier_bundle, test_items)
                classifier_result.update(common_run_metadata)
                classifier_result.update({
                    "embedder": embedder_info,
                    "classifier_fit_count": len(seed_train_items),
                })
                results_by_arm["query_classifier"].append(classifier_result)
                save_json(args.output_dir / f"seed_{seed}" / "query_classifier.json", classifier_result)
                progress.log("arm_done", seed=seed, arm="query_classifier", accuracy=classifier_result["accuracy"], recall_at_5=classifier_result["recall_at_5"])

            progress.log("seed_complete", seed=seed, arm_count=sum(1 for arm in arms if arm in results_by_arm and results_by_arm[arm]))
            if args.self_test:
                progress.log("self_test_complete", seed=seed, arm_count=sum(1 for arm in arms if arm in results_by_arm and results_by_arm[arm]))
                break

        combined_summary = {
            "paper_eligible": args.data_mode == "toolbench_train",
            "config": {
                "groups": groups,
                "arms": arms,
                "seeds": seeds,
                "data_mode": args.data_mode,
                "partition_mode": args.partition_mode,
                "client_count": args.client_count,
                "top_k": args.top_k,
                "retrieval_pool_size": args.retrieval_pool_size,
                "retrieval_mode": args.retrieval_mode,
                "reranker_variant": args.reranker_variant,
                "merge_policy": args.merge_policy,
                "junk_filter": args.junk_filter,
                "build_only": args.build_only,
                "embed_model": args.embed_model,
                "local_embedder": embedder_info,
                "git_commit": commit,
                "allow_dirty": args.allow_dirty,
                "dirty_entry_count": len(dirty_entries),
                "dirty_entries_sample": dirty_entries[:20],
            },
            "arms": {arm: aggregate_seed_summaries(seed_summaries) for arm, seed_summaries in results_by_arm.items() if seed_summaries},
            "synapse_hashes": synapse_hashes,
        }
        save_json(args.output_dir / "combined_summary.json", combined_summary)
        progress.log("complete", output_dir=str(args.output_dir), arm_count=len(combined_summary["arms"]))


if __name__ == "__main__":
    main()
