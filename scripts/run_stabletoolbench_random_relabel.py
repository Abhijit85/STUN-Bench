#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_ROOT = Path(os.environ.get("FEDRAG_CANONICAL_ROOT", "<REPO_ROOT>")).resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from math_qa import JinaAIClient
from scripts.run_stabletoolbench_federated import (
    DEFAULT_MODEL_PATH,
    DEFAULT_STB_ROOT,
    ProgressLogger,
    assert_clean_tree,
    evaluate_reranker_arm,
    hash_package,
    load_local_backend,
    load_package_file,
    resolve_local_embedder,
    save_json,
    temporary_env,
)
from scripts.run_stabletoolbench_oracle import evaluate_oracle_arm
from scripts.run_stabletoolbench_typing_isolation import load_eval_queries, load_seed_packages
from synapse.knowledge.compendium import KnowledgeArtifact, KnowledgePackage


DEFAULT_OUTPUT_DIR = CANONICAL_ROOT / "artifacts" / "results" / "stabletoolbench_random_relabel_seed456_r1"
DEFAULT_PACKAGE_ROOT = CANONICAL_ROOT / "artifacts" / "verification" / "stabletoolbench_clean_anchor_seed456_r4" / "packages"
DEFAULT_SYNAPSE_PACKAGE = DEFAULT_PACKAGE_ROOT / "seed_456" / "renderswap_cache" / "typed_conflictlog_conflict_00.json"
DEFAULT_SOURCE_CELL = CANONICAL_ROOT / "artifacts" / "results" / "stabletoolbench_2x2_seed456_r2" / "typed_conflictlog" / "conflict_00" / "seed_456.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="StableToolBench random-relabel control over seed-456 typed compendium.")
    parser.add_argument("--seed", type=int, default=456)
    parser.add_argument("--source-cell", type=Path, default=DEFAULT_SOURCE_CELL)
    parser.add_argument("--synapse-package", type=Path, default=DEFAULT_SYNAPSE_PACKAGE)
    parser.add_argument("--packages-root", type=Path, default=DEFAULT_PACKAGE_ROOT)
    parser.add_argument("--groups", type=str, default="G1_instruction,G2_instruction")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retrieval-pool-size", type=int, default=20)
    parser.add_argument("--oracle-retrieval-pool-size", type=int, default=5)
    parser.add_argument("--retrieval-mode", type=str, default="distinct_tool_topk")
    parser.add_argument("--reranker-variant", type=str, default="V3")
    parser.add_argument("--embed-model", type=str, default="jina-embeddings-v2-base-en")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--stb-root", type=Path, default=DEFAULT_STB_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def relabel_experience(package: KnowledgePackage, seed: int) -> tuple[KnowledgePackage, dict[str, Any]]:
    tools = sorted({
        str((artifact.metadata or {}).get("tool") or "").strip()
        for artifact in package.artifacts
        if (artifact.metadata or {}).get("artifact_origin") == "client_experience"
    })
    tools = [tool for tool in tools if tool]
    shuffled = list(tools)
    rng = random.Random(seed)
    for _ in range(16):
        rng.shuffle(shuffled)
        if all(left != right for left, right in zip(tools, shuffled)):
            break
    mapping = dict(zip(tools, shuffled))
    if any(tool == mapping.get(tool) for tool in tools) and len(tools) > 1:
        shuffled = shuffled[1:] + shuffled[:1]
        mapping = dict(zip(tools, shuffled))

    changed = 0
    artifacts: list[KnowledgeArtifact] = []
    for artifact in copy.deepcopy(package.artifacts):
        metadata = dict(artifact.metadata or {})
        if metadata.get("artifact_origin") != "client_experience":
            artifacts.append(artifact)
            continue
        old_tool = str(metadata.get("tool") or "").strip()
        new_tool = mapping.get(old_tool)
        if not new_tool:
            artifacts.append(artifact)
            continue
        metadata["original_tool_before_random_relabel"] = old_tool
        metadata["tool"] = new_tool
        metadata["domain"] = new_tool
        metadata["scenario"] = new_tool
        artifacts.append(
            KnowledgeArtifact(
                signature=artifact.signature,
                text=artifact.text,
                structured_payload=artifact.structured_payload,
                metadata=metadata,
                textgrad_variable=artifact.textgrad_variable,
            )
        )
        changed += 1

    relabeled = KnowledgePackage(
        source_id=f"{package.source_id}_random_relabel",
        artifacts=artifacts,
        metadata={**dict(package.metadata or {}), "random_relabel_seed": seed, "random_relabel_tool_count": len(mapping)},
    )
    return relabeled, {
        "random_relabel_seed": seed,
        "mapping_sha256": hashlib.sha256(json.dumps(mapping, sort_keys=True).encode("utf-8")).hexdigest(),
        "tool_count": len(mapping),
        "artifact_count_relabelled": changed,
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    commit, dirty = assert_clean_tree(allow_dirty=args.allow_dirty)
    progress = ProgressLogger(args.output_dir)
    progress.log("launch", git_commit=commit, dirty_entry_count=len(dirty))

    embedder = resolve_local_embedder()
    jina_client = JinaAIClient(api_keys=os.environ.get("JINA_API_KEY") and [os.environ["JINA_API_KEY"]] or [])
    source_cell = load_json(args.source_cell)
    synapse_package, synapse_hash = load_package_file(args.synapse_package)
    if synapse_hash != source_cell["compendium"]["global_sha256"]:
        raise RuntimeError(
            f"source/package hash mismatch: source={source_cell['compendium']['global_sha256']} package={synapse_hash}"
        )
    _client_packages, tool_doc_package, tool_doc_hash = load_seed_packages(args.packages_root, args.seed)
    flat_package, flat_hash = load_package_file(args.packages_root / f"seed_{args.seed}" / "flat_pool.json")
    groups = parse_csv(args.groups)
    test_items = load_eval_queries(args.stb_root, groups, tool_doc_package)
    relabeled_package, relabel_meta = relabel_experience(synapse_package, args.seed)
    progress.log(
        "setup_done",
        seed=args.seed,
        query_count=len(test_items),
        source_cell=str(args.source_cell),
        source_cell_sha256=sha256_file(args.source_cell),
        synapse_sha256=synapse_hash,
        relabeled_sha256=hash_package(relabeled_package),
        **relabel_meta,
    )

    results: dict[str, Any] = {}
    with temporary_env({
        "JINA_LOCAL_EMBED_MODEL": embedder["model_path"],
        "JINA_LOCAL_EMBED_DEVICE": embedder["device"],
        "JINA_LOCAL_EMBED_LOCAL_ONLY": embedder["local_only"],
        "JINA_API_KEY": None,
    }):
        backend = load_local_backend(args.model_path)
        specs = [
            ("synapse_unpermuted_real", synapse_package, False),
            ("synapse_random_relabel_real", relabeled_package, False),
            ("docs_only_real", flat_package, False),
            ("synapse_unpermuted_oracle", synapse_package, True),
            ("synapse_random_relabel_oracle", relabeled_package, True),
            ("docs_only_oracle", flat_package, True),
        ]
        for name, package, oracle in specs:
            progress.log("arm_begin", arm=name, oracle_retrieval=oracle, package_sha256=hash_package(package))
            if oracle:
                result = evaluate_oracle_arm(
                    name,
                    package,
                    test_items,
                    jina_client,
                    args.embed_model,
                    backend,
                    args.top_k,
                    args.oracle_retrieval_pool_size,
                    args.retrieval_mode,
                    args.reranker_variant,
                    progress=progress,
                    seed=args.seed,
                )
            else:
                result = evaluate_reranker_arm(
                    name,
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
                    seed=args.seed,
                )
            result.update({
                "seed": args.seed,
                "arm": name,
                "oracle_retrieval": oracle,
                "random_relabel": "random_relabel" in name,
                "source_cell_path": str(args.source_cell),
                "source_cell_sha256": sha256_file(args.source_cell),
                "source_compendium_sha256": source_cell["compendium"]["global_sha256"],
                "package_sha256": hash_package(package),
                "tool_doc_sha256": tool_doc_hash,
                "flat_pool_sha256": flat_hash,
                "relabel_meta": relabel_meta if "random_relabel" in name else None,
                "embedder": embedder,
                "git_commit": commit,
                "paper_eligible": True,
                "metric_definition": {
                    "correct": "predicted_tool in gold_parent_tools",
                    "recall_at_5": "any gold tool present among candidate tools",
                },
            })
            out_path = args.output_dir / f"{name}.json"
            save_json(out_path, result)
            results[name] = {k: result[k] for k in ("accuracy", "recall_at_5") if k in result}
            if oracle:
                results[name]["oracle_insertion_fraction"] = result["oracle_insertion_fraction"]
            progress.log("arm_done", arm=name, accuracy=result["accuracy"], recall_at_5=result["recall_at_5"])

    summary = {
        "paper_eligible": True,
        "config": {
            "git_commit": commit,
            "seed": args.seed,
            "groups": groups,
            "source_cell": str(args.source_cell),
            "source_cell_sha256": sha256_file(args.source_cell),
            "source_compendium_sha256": source_cell["compendium"]["global_sha256"],
            "synapse_package_path": str(args.synapse_package),
            "synapse_package_sha256": synapse_hash,
            "relabeled_package_sha256": hash_package(relabeled_package),
            "relabel_meta": relabel_meta,
            "retrieval_mode": args.retrieval_mode,
            "retrieval_pool_size": args.retrieval_pool_size,
            "oracle_retrieval_pool_size": args.oracle_retrieval_pool_size,
            "top_k": args.top_k,
            "reranker_variant": args.reranker_variant,
            "embedder": embedder,
        },
        "results": results,
    }
    save_json(args.output_dir / "summary.json", summary)
    progress.log("complete", output_dir=str(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
