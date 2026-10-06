#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from synapse.edge.aggregator import EdgeAggregator, EdgeConfig
from synapse.knowledge.compendium import KnowledgeArtifact, KnowledgePackage
from scripts.run_stabletoolbench_federated import (
    DEFAULT_MODEL_PATH,
    DEFAULT_STB_ROOT,
    ProgressLogger,
    assert_clean_tree,
    evaluate_reranker_arm,
    filter_eval_queries,
    hash_package,
    load_package_file,
    load_stabletoolbench_queries,
    load_local_backend,
    merge_heartbeat,
    normalize_query_text,
    read_edge_conflict_log,
    save_json,
    temporary_env,
    tool_description,
    resolve_local_embedder,
)
from scripts.run_stabletoolbench_federated import ToolDoc as RunnerToolDoc
from math_qa import JinaAIClient


SUPPORTED_ARMS = ("typed_conflictlog", "typed_round_delayed", "typed_majority", "flat_majority")


@dataclass(frozen=True)
class ArmSpec:
    name: str
    payload_mode: str
    merge_policy: str


ARM_SPECS = {
    "typed_conflictlog": ArmSpec("typed_conflictlog", "typed", "conflict_log"),
    "typed_round_delayed": ArmSpec("typed_round_delayed", "typed", "round_delayed"),
    "typed_majority": ArmSpec("typed_majority", "typed", "majority"),
    "flat_majority": ArmSpec("flat_majority", "flat_json", "majority"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="StableToolBench 2x2 typing-isolation replay over persisted packages.")
    parser.add_argument("--load-packages-root", type=Path, required=True)
    parser.add_argument("--source-data-mode", type=str, required=True, choices=["stable_holdout", "toolbench_train"])
    parser.add_argument("--groups", type=str, default="G1_instruction,G2_instruction")
    parser.add_argument("--arms", type=str, default=",".join(SUPPORTED_ARMS))
    parser.add_argument("--contradiction-rates", type=str, default="0,20,40,60")
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--retrieval-mode", type=str, default="distinct_tool_topk")
    parser.add_argument("--reranker-variant", type=str, default="V3")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retrieval-pool-size", type=int, default=20)
    parser.add_argument("--embed-model", type=str, default="jina-embeddings-v2-base-en")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--stb-root", type=Path, default=DEFAULT_STB_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args()


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_int_csv(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def flat_json_payload(metadata: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    compact = {
        "scenario": metadata.get("scenario"),
        "tool_description": payload.get("tool_description"),
        "scenario_context": payload.get("scenario_context"),
        "precautions": payload.get("precautions"),
        "annex_summary": payload.get("annex_summary"),
        "conflict_log": payload.get("conflict_log"),
    }
    compact = {key: value for key, value in compact.items() if value not in (None, "", [], {})}
    return {
        "payload_mode": "flat_json",
        "serialized_payload": json.dumps(compact, sort_keys=True, ensure_ascii=True),
    }


def flatten_package(package: KnowledgePackage) -> KnowledgePackage:
    flattened: list[KnowledgeArtifact] = []
    for artifact in package.artifacts:
        payload = artifact.structured_payload or {}
        metadata = dict(artifact.metadata or {})
        flat_payload = flat_json_payload(metadata, payload if isinstance(payload, dict) else {})
        text = flat_payload["serialized_payload"]
        flattened.append(
            KnowledgeArtifact(
                signature=artifact.signature,
                text=text,
                structured_payload=flat_payload,
                metadata=metadata,
                textgrad_variable=artifact.textgrad_variable,
            )
        )
    return KnowledgePackage(source_id=package.source_id, artifacts=flattened, metadata=dict(package.metadata or {}))


def package_registry(tool_doc_package: KnowledgePackage) -> dict[str, RunnerToolDoc]:
    registry: dict[str, RunnerToolDoc] = {}
    for artifact in tool_doc_package.artifacts:
        tool = str((artifact.metadata or {}).get("tool") or "").strip()
        if not tool:
            continue
        payload = artifact.structured_payload or {}
        api_names: list[str] = []
        descriptions: list[str] = []
        tool_desc = str(payload.get("tool_description") or "").strip()
        if tool_desc:
            parts = [part.strip() for part in tool_desc.split(";")]
            for part in parts:
                if part.startswith("apis="):
                    api_names = [item.strip() for item in part.removeprefix("apis=").split(",") if item.strip()]
                if part.startswith("docs="):
                    desc = part.removeprefix("docs=").strip()
                    if desc:
                        descriptions.append(desc)
        if not descriptions:
            scenario_context = str(payload.get("scenario_context") or "").strip()
            if scenario_context:
                descriptions.append(scenario_context)
        category = str((artifact.metadata or {}).get("category") or "unknown")
        registry[tool] = RunnerToolDoc(tool_name=tool, categories=[category], api_names=api_names, descriptions=descriptions)
    return registry


def load_seed_packages(root: Path, seed: int) -> tuple[dict[str, KnowledgePackage], KnowledgePackage, str]:
    seed_dir = root / f"seed_{seed}" if (root / f"seed_{seed}").exists() else root
    client_packages: dict[str, KnowledgePackage] = {}
    for path in sorted(seed_dir.glob("client_*.json")):
        package, _sha = load_package_file(path)
        client_packages[path.stem] = package
    if not client_packages:
        raise FileNotFoundError(f"No client packages found under {seed_dir}")
    tool_doc_package, tool_doc_hash = load_package_file(seed_dir / "tool_docs.json")
    return client_packages, tool_doc_package, tool_doc_hash


def fingerprint(rate: int, seed: int, signature: str) -> bool:
    if rate <= 0:
        return False
    digest = hashlib.sha256(f"{seed}:{signature}".encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) % 100) < rate


def choose_neighbors(client_packages: dict[str, KnowledgePackage]) -> dict[str, KnowledgeArtifact]:
    by_tool: dict[str, list[KnowledgeArtifact]] = defaultdict(list)
    for package in client_packages.values():
        for artifact in package.artifacts:
            payload = artifact.structured_payload or {}
            if not isinstance(payload, dict) or payload.get("type") != "usage_scenario":
                continue
            tool = str((artifact.metadata or {}).get("tool") or "").strip()
            if tool:
                by_tool[tool].append(artifact)

    neighbors: dict[str, KnowledgeArtifact] = {}

    def score(left: KnowledgeArtifact, right: KnowledgeArtifact) -> tuple[int, int]:
        left_text = str((left.structured_payload or {}).get("scenario_context") or "")
        right_text = str((right.structured_payload or {}).get("scenario_context") or "")
        left_tokens = set(normalize_query_text(left_text).split())
        right_tokens = set(normalize_query_text(right_text).split())
        overlap = len(left_tokens & right_tokens)
        return overlap, -abs(len(left_text) - len(right_text))

    for artifacts in by_tool.values():
        for artifact in artifacts:
            choices = [candidate for candidate in artifacts if candidate.signature != artifact.signature]
            if choices:
                neighbors[artifact.signature] = max(choices, key=lambda candidate: score(artifact, candidate))
    return neighbors


def inject_contradictions(client_packages: dict[str, KnowledgePackage], rate: int, seed: int) -> tuple[dict[str, KnowledgePackage], dict[str, Any]]:
    mutated = copy.deepcopy(client_packages)
    original_lookup = {
        artifact.signature: artifact
        for package in client_packages.values()
        for artifact in package.artifacts
    }
    neighbors = choose_neighbors(client_packages)
    contradicted: list[str] = []
    for client_id, package in mutated.items():
        updated_artifacts: list[KnowledgeArtifact] = []
        for artifact in package.artifacts:
            payload = dict(artifact.structured_payload or {})
            if payload.get("type") == "usage_scenario" and artifact.signature in neighbors and fingerprint(rate, seed, artifact.signature):
                neighbor = neighbors[artifact.signature]
                neighbor_payload = neighbor.structured_payload or {}
                neighbor_context = str(neighbor_payload.get("scenario_context") or "").strip()
                if neighbor_context:
                    payload["scenario_context"] = neighbor_context
                    payload["contradicted"] = True
                    payload["contradiction_source"] = neighbor.signature
                    contradicted.append(artifact.signature)
                    artifact = KnowledgeArtifact(
                        signature=artifact.signature,
                        text=artifact.text,
                        structured_payload=payload,
                        metadata=dict(artifact.metadata or {}),
                        textgrad_variable=artifact.textgrad_variable,
                    )
            updated_artifacts.append(artifact)
        mutated[client_id] = KnowledgePackage(source_id=package.source_id, artifacts=updated_artifacts, metadata=dict(package.metadata or {}))
    contradicted = sorted(set(contradicted))
    return mutated, {
        "rate": rate,
        "seed": seed,
        "contradiction_hash": hashlib.sha256("\n".join(contradicted).encode("utf-8")).hexdigest(),
        "contradicted_signatures": contradicted,
        "contradicted_count": len(contradicted),
        "eligible_count": sum(1 for signature in original_lookup if signature in neighbors),
    }


def merge_for_arm(client_packages: dict[str, KnowledgePackage], tool_doc_package: KnowledgePackage, arm: ArmSpec, seed: int) -> tuple[KnowledgePackage, dict[str, Any]]:
    if arm.payload_mode == "flat_json":
        client_packages = {client_id: flatten_package(package) for client_id, package in client_packages.items()}
    with temporary_env({"SYNAPSE_EDGE_MERGE_POLICY": arm.merge_policy}):
        aggregator = EdgeAggregator(EdgeConfig(edge_id=f"stabletoolbench_typing_seed_{seed}_{arm.name}"))
        merged = aggregator.merge_packages(list(client_packages.values()))
    if merged is None:
        raise RuntimeError(f"{arm.name} merge returned no package")
    package = KnowledgePackage(
        source_id=f"{arm.name}_with_docs",
        artifacts=list(tool_doc_package.artifacts) + list(merged.artifacts),
        metadata={"sources": ["tool_docs", arm.name]},
    )
    return package, {
        "global_sha256": hash_package(package),
        "artifact_count": len(package.artifacts),
        "edge_conflict_log": read_edge_conflict_log(aggregator),
    }


def subset_groups(rows: list[TestQueryLike], groups: set[str]) -> list[TestQueryLike]:
    return [row for row in rows if row.group in groups]


class TestQueryLike:
    def __init__(self, query_id: str, query: str, gold_tools: list[str], group: str, categories: list[str], api_list: list[dict[str, Any]]):
        self.query_id = query_id
        self.query = query
        self.gold_tools = gold_tools
        self.group = group
        self.categories = categories
        self.api_list = api_list


def load_eval_queries(stb_root: Path, groups: list[str], tool_doc_package: KnowledgePackage) -> list[TestQueryLike]:
    raw = load_stabletoolbench_queries(stb_root, groups)
    registry = package_registry(tool_doc_package)
    filtered, _info = filter_eval_queries(raw, registry)
    return [TestQueryLike(item.query_id, item.query, item.gold_tools, item.group, item.categories, item.api_list) for item in filtered]


def aggregate_runs(raw_runs: dict[str, dict[int, dict[int, dict[str, Any]]]], arms: list[str], rates: list[int], seeds: list[int]) -> dict[str, Any]:
    per_arm: dict[str, dict[str, Any]] = {}
    for arm in arms:
        per_arm[arm] = {}
        for rate in rates:
            per_seed = {str(seed): float(raw_runs[arm][rate][seed]["accuracy"]) for seed in seeds}
            values = list(per_seed.values())
            per_arm[arm][str(rate)] = {
                "mean": statistics.mean(values) if values else 0.0,
                "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
                "per_seed": per_seed,
            }
    return per_arm


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    commit, dirty_entries = assert_clean_tree(allow_dirty=args.allow_dirty)
    progress = ProgressLogger(args.output_dir)
    progress.log("launch", git_commit=commit, dirty_entry_count=len(dirty_entries))

    arms = parse_csv(args.arms)
    unknown = sorted(set(arms) - set(ARM_SPECS))
    if unknown:
        raise ValueError(f"Unknown arms: {', '.join(unknown)}")
    rates = parse_int_csv(args.contradiction_rates)
    seeds = parse_int_csv(args.seeds)
    groups = parse_csv(args.groups)
    embedder_info = resolve_local_embedder()
    jina_client = JinaAIClient(api_keys=os.environ.get("JINA_API_KEY") and [os.environ["JINA_API_KEY"]] or [])
    progress.log("init_embedding_client_done", embed_model=args.embed_model, local_model=embedder_info["model_path"])

    with temporary_env({
        "JINA_LOCAL_EMBED_MODEL": embedder_info["model_path"],
        "JINA_LOCAL_EMBED_DEVICE": embedder_info["device"],
        "JINA_LOCAL_EMBED_LOCAL_ONLY": embedder_info["local_only"],
        "JINA_API_KEY": None,
    }):
        progress.log("load_backend_begin", model_path=args.model_path)
        backend = load_local_backend(args.model_path)
        progress.log("load_backend_done")
        raw_runs: dict[str, dict[int, dict[int, dict[str, Any]]]] = {arm: {rate: {} for rate in rates} for arm in arms}
        contradiction_hashes: dict[int, dict[int, dict[str, Any]]] = {rate: {} for rate in rates}

        for seed in seeds:
            progress.log("seed_begin", seed=seed)
            client_packages, tool_doc_package, tool_doc_hash = load_seed_packages(args.load_packages_root, seed)
            test_items = load_eval_queries(args.stb_root, groups, tool_doc_package)
            progress.log("seed_inputs_ready", seed=seed, test_count=len(test_items), client_count=len(client_packages), tool_doc_sha256=tool_doc_hash)
            for rate in rates:
                contradicted_packages, contradiction_info = inject_contradictions(client_packages, rate, seed)
                contradiction_hashes[rate][seed] = contradiction_info
                progress.log("contradiction_set_ready", seed=seed, rate=rate, contradiction_hash=contradiction_info["contradiction_hash"], contradicted_count=contradiction_info["contradicted_count"])
                for arm_name in arms:
                    arm = ARM_SPECS[arm_name]
                    progress.log("arm_begin", seed=seed, rate=rate, arm=arm_name)
                    with merge_heartbeat(progress, seed=seed, arm=arm_name, merge_policy=arm.merge_policy):
                        package, merge_info = merge_for_arm(contradicted_packages, tool_doc_package, arm, seed)
                    result = evaluate_reranker_arm(
                        arm_name,
                        package,
                        test_items,
                        jina_client,
                        args.embed_model,
                        backend,
                        args.top_k,
                        args.retrieval_pool_size,
                        args.retrieval_mode,
                        args.reranker_variant,
                        progress=progress,
                        seed=seed,
                    )
                    result.update({
                        "seed": seed,
                        "rate": rate,
                        "arm": arm_name,
                        "git_commit": commit,
                        "groups": groups,
                        "contradiction_hash": contradiction_info["contradiction_hash"],
                        "contradicted_signatures": contradiction_info["contradicted_signatures"],
                        "contradicted_count": contradiction_info["contradicted_count"],
                        "compendium": merge_info,
                        "tool_doc_sha256": tool_doc_hash,
                        "merge_policy": arm.merge_policy,
                        "payload_mode": arm.payload_mode,
                        "embedder": embedder_info,
                        "source_data_mode": args.source_data_mode,
                        "paper_eligible": args.source_data_mode == "toolbench_train",
                    })
                    raw_runs[arm_name][rate][seed] = result
                    out_path = args.output_dir / arm_name / f"conflict_{rate:02d}" / f"seed_{seed}.json"
                    save_json(out_path, result)
                    reference = contradiction_hashes[rate][seed]["contradiction_hash"]
                    if result["contradiction_hash"] != reference:
                        raise RuntimeError(
                            f"Injected-set hash mismatch at rate={rate} seed={seed} arm={arm_name}: "
                            f"expected {reference} got {result['contradiction_hash']}"
                        )
                    progress.log("arm_done", seed=seed, rate=rate, arm=arm_name, accuracy=result["accuracy"], recall_at_5=result["recall_at_5"])
            progress.log("seed_complete", seed=seed)

        summary = {
            "paper_eligible": args.source_data_mode == "toolbench_train",
            "config": {
                "git_commit": commit,
                "groups": groups,
                "arms": arms,
                "rates": rates,
                "seeds": seeds,
                "retrieval_mode": args.retrieval_mode,
                "retrieval_pool_size": args.retrieval_pool_size,
                "top_k": args.top_k,
                "reranker_variant": args.reranker_variant,
                "model_path": args.model_path,
                "load_packages_root": str(args.load_packages_root),
                "source_data_mode": args.source_data_mode,
                "embed_model": args.embed_model,
                "local_embedder": embedder_info,
            },
            "contradiction_hashes": {
                str(rate): {str(seed): contradiction_hashes[rate][seed] for seed in seeds}
                for rate in rates
            },
            "per_arm": aggregate_runs(raw_runs, arms, rates, seeds),
        }
        save_json(args.output_dir / "summary.json", summary)
        progress.log("complete", output_dir=str(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
