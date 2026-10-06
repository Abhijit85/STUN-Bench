#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_ROOT = Path(os.environ.get('FEDRAG_CANONICAL_ROOT', '<REPO_ROOT>')).resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from math_qa import JinaAIClient
from scripts.run_stabletoolbench_federated import (
    DEFAULT_MODEL_PATH,
    DEFAULT_STB_ROOT,
    ProgressLogger,
    assert_clean_tree,
    evaluate_reranker_arm,
    load_local_backend,
    merge_heartbeat,
    save_json,
    temporary_env,
    resolve_local_embedder,
    load_package_file,
    render_tool_doc_text,
    render_usage_scenario_text,
    hash_package,
    save_package,
)
from scripts.run_stabletoolbench_typing_isolation import (
    ARM_SPECS,
    load_eval_queries,
    load_seed_packages,
    inject_contradictions,
    merge_for_arm,
    flatten_package,
)
from synapse.knowledge.compendium import KnowledgeArtifact, KnowledgePackage

DEFAULT_SOURCE_SUMMARIES = {
    42: CANONICAL_ROOT / "artifacts" / "results" / "stabletoolbench_2x2_seed42_r3b" / "summary.json",
    123: Path("<REPO_ROOT>_runs/artifacts/results/stabletoolbench_2x2_seed123_r5/summary.json"),
    456: CANONICAL_ROOT / "artifacts" / "results" / "stabletoolbench_2x2_seed456_r2" / "summary.json",
}
DEFAULT_OUTPUT_DIR = CANONICAL_ROOT / "artifacts" / "results" / "stabletoolbench_2x2_renderswap_r1"
ARM_SOURCES = {
    "typed_conflictlog": "typed_conflictlog",
    "typed_merge_flat_render": "typed_conflictlog",
    "flat_majority": "flat_majority",
    "flat_merge_structured_render": "flat_majority",
}
ARM_RENDER_MODE = {
    "typed_conflictlog": "structured",
    "typed_merge_flat_render": "flat_json",
    "flat_majority": "flat_json",
    "flat_merge_structured_render": "structured",
}
ARM_MERGE_MODE = {
    "typed_conflictlog": "typed_conflictlog",
    "typed_merge_flat_render": "typed_conflictlog",
    "flat_majority": "flat_majority",
    "flat_merge_structured_render": "flat_majority",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="StableToolBench render-swap replay over saved 2x2 cells.")
    parser.add_argument("--source-summary", action="append", default=[], help="Override source summary path(s); one per seed or repeated.")
    parser.add_argument("--seeds", type=str, default="42,123,456")
    parser.add_argument("--rates", type=str, default="0,60")
    parser.add_argument("--arms", type=str, default="typed_conflictlog,flat_majority,typed_merge_flat_render,flat_merge_structured_render")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retrieval-pool-size", type=int, default=20)
    parser.add_argument("--retrieval-mode", type=str, default="distinct_tool_topk")
    parser.add_argument("--reranker-variant", type=str, default="V3")
    parser.add_argument("--embed-model", type=str, default="jina-embeddings-v2-base-en")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--stb-root", type=Path, default=DEFAULT_STB_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--skip-baseline-replay-check",
        action="store_true",
        help="Allow baseline replay metrics to differ from source cells, e.g. when intentionally swapping reranker models.",
    )
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args()


def parse_csv_int(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_source_summaries(args: argparse.Namespace, seeds: list[int]) -> dict[int, Path]:
    if not args.source_summary:
        return {seed: DEFAULT_SOURCE_SUMMARIES[seed] for seed in seeds}
    provided = [Path(item) for item in args.source_summary]
    if len(provided) == 1:
        return {seed: provided[0] for seed in seeds}
    if len(provided) != len(seeds):
        raise ValueError("--source-summary must be omitted, passed once, or passed once per seed")
    return {seed: path for seed, path in zip(seeds, provided)}


def source_cell_path(summary_path: Path, arm: str, rate: int, seed: int) -> Path:
    return summary_path.parent / arm / f"conflict_{rate:02d}" / f"seed_{seed}.json"


def relabel_as_flat(package: KnowledgePackage) -> KnowledgePackage:
    return flatten_package(package)


def decode_flat_payload(package: KnowledgePackage) -> tuple[KnowledgePackage, int]:
    repaired: list[KnowledgeArtifact] = []
    decode_failures = 0
    for artifact in package.artifacts:
        payload = artifact.structured_payload or {}
        serialized = str(payload.get("serialized_payload") or "")
        metadata = dict(artifact.metadata or {})
        if not serialized:
            decode_failures += 1
            repaired.append(artifact)
            continue
        decoded = json.loads(serialized)
        roundtrip = json.dumps(decoded, sort_keys=True, ensure_ascii=True)
        if roundtrip != serialized:
            decode_failures += 1
        payload_type = "tool_doc" if metadata.get("artifact_origin") == "tool_doc" else "usage_scenario"
        typed_payload = {
            "type": payload_type,
            "payload_mode": "typed",
            "tool_description": decoded.get("tool_description") or "",
            "scenario_context": decoded.get("scenario_context") or "",
            "precautions": list(decoded.get("precautions") or []),
            "annex_summary": decoded.get("annex_summary") or "",
            "conflict_log": list(decoded.get("conflict_log") or []),
        }
        tool = str(metadata.get("tool") or metadata.get("domain") or "")
        if payload_type == "tool_doc":
            descriptions = [typed_payload["scenario_context"]] if typed_payload["scenario_context"] else []
            apis = []
            tool_description = str(typed_payload["tool_description"] or "")
            for part in tool_description.split(";"):
                part = part.strip()
                if part.startswith("apis="):
                    apis = [item.strip() for item in part.split("=", 1)[1].split(",") if item.strip()]
                if part.startswith("docs="):
                    desc = part.split("=", 1)[1].strip()
                    if desc:
                        descriptions = [desc]
            doc_like = type("DocLike", (), {"categories": [str(metadata.get("category") or "unknown")], "api_names": apis, "descriptions": descriptions})
            text = render_tool_doc_text(tool, doc_like)
        else:
            text = render_usage_scenario_text(tool, typed_payload)
        repaired.append(
            KnowledgeArtifact(
                signature=artifact.signature,
                text=text,
                structured_payload=typed_payload,
                metadata=metadata,
                textgrad_variable=artifact.textgrad_variable,
            )
        )
    return KnowledgePackage(source_id=package.source_id, artifacts=repaired, metadata=dict(package.metadata or {})), decode_failures


def rebuild_source_package(packages_root: Path, seed: int, source_arm: str, rate: int, progress: ProgressLogger | None = None) -> tuple[KnowledgePackage, dict[str, Any], str]:
    seed_dir = packages_root / f"seed_{seed}" if (packages_root / f"seed_{seed}").exists() else packages_root
    cache_dir = seed_dir / 'renderswap_cache'
    cache_path = cache_dir / f"{source_arm}_conflict_{rate:02d}.json"
    if cache_path.exists():
        package, package_hash = load_package_file(cache_path)
        merge_info = {"global_sha256": package_hash, "artifact_count": len(package.artifacts), "edge_conflict_log": []}
        tool_doc_hash = ''
        if progress is not None:
            progress.log("source_package_cache_hit", seed=seed, rate=rate, arm=source_arm, path=str(cache_path), artifact_count=len(package.artifacts))
        return package, merge_info, tool_doc_hash
    if progress is not None:
        progress.log("source_package_build_begin", seed=seed, rate=rate, arm=source_arm, packages_root=str(packages_root))
    client_packages, tool_doc_package, tool_doc_hash = load_seed_packages(packages_root, seed)
    if progress is not None:
        progress.log("source_package_loaded", seed=seed, rate=rate, arm=source_arm, client_count=len(client_packages), tool_doc_artifacts=len(tool_doc_package.artifacts))
    contradicted_packages, _contradiction_info = inject_contradictions(client_packages, rate, seed)
    if progress is not None:
        progress.log("source_package_contradictions_done", seed=seed, rate=rate, arm=source_arm, contradicted_count=int(_contradiction_info.get('contradicted_count', 0)))
    if progress is not None:
        with merge_heartbeat(progress, seed=seed, arm=source_arm, merge_policy=ARM_SPECS[source_arm].merge_policy):
            package, merge_info = merge_for_arm(contradicted_packages, tool_doc_package, ARM_SPECS[source_arm], seed)
    else:
        package, merge_info = merge_for_arm(contradicted_packages, tool_doc_package, ARM_SPECS[source_arm], seed)
    cache_dir.mkdir(parents=True, exist_ok=True)
    save_package(cache_path, package, merge_info['global_sha256'])
    if progress is not None:
        progress.log("source_package_build_done", seed=seed, rate=rate, arm=source_arm, path=str(cache_path), artifact_count=len(package.artifacts))
    return package, merge_info, tool_doc_hash


def aggregate(results: list[dict[str, Any]], arms: list[str], rates: list[int]) -> dict[str, Any]:
    by_arm: dict[str, dict[str, Any]] = {}
    for arm in arms:
        arm_rows = [row for row in results if row["arm"] == arm]
        by_rate: dict[str, Any] = {}
        for rate in rates:
            subset = [row for row in arm_rows if row["rate"] == rate]
            accs = [float(row["accuracy"]) for row in subset]
            recs = [float(row["recall_at_5"]) for row in subset]
            by_rate[str(rate)] = {
                "mean_accuracy": statistics.mean(accs) if accs else 0.0,
                "sd_accuracy": statistics.stdev(accs) if len(accs) > 1 else 0.0,
                "mean_recall_at_5": statistics.mean(recs) if recs else 0.0,
                "sd_recall_at_5": statistics.stdev(recs) if len(recs) > 1 else 0.0,
                "per_seed": {str(row["seed"]): {"accuracy": row["accuracy"], "recall_at_5": row["recall_at_5"]} for row in subset},
            }
        by_arm[arm] = by_rate
    return by_arm


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    commit, dirty = assert_clean_tree(allow_dirty=args.allow_dirty)
    progress = ProgressLogger(args.output_dir)
    progress.log("launch", git_commit=commit, dirty_entry_count=len(dirty))

    seeds = parse_csv_int(args.seeds)
    rates = parse_csv_int(args.rates)
    arms = parse_csv(args.arms)
    source_summaries = resolve_source_summaries(args, seeds)
    embedder = resolve_local_embedder()
    jina_client = JinaAIClient(api_keys=os.environ.get("JINA_API_KEY") and [os.environ["JINA_API_KEY"]] or [])
    progress.log("init_embedding_client_done", local_model=embedder["model_path"], embed_model=args.embed_model)

    with temporary_env({
        "JINA_LOCAL_EMBED_MODEL": embedder["model_path"],
        "JINA_LOCAL_EMBED_DEVICE": embedder["device"],
        "JINA_LOCAL_EMBED_LOCAL_ONLY": embedder["local_only"],
        "JINA_API_KEY": None,
    }):
        backend = load_local_backend(args.model_path)
        all_results: list[dict[str, Any]] = []
        decode_failures_by_cell: dict[str, int] = {}
        for seed in seeds:
            summary_path = source_summaries[seed]
            summary = load_json(summary_path)
            groups = summary["config"]["groups"]
            packages_root = Path(summary["config"]["load_packages_root"])
            if not packages_root.is_absolute():
                packages_root = (CANONICAL_ROOT / packages_root).resolve()
            _client_packages, tool_doc_package, _tool_doc_hash = load_seed_packages(packages_root, seed)
            test_items = load_eval_queries(args.stb_root, groups, tool_doc_package)
            progress.log("seed_begin", seed=seed, packages_root=str(packages_root), query_count=len(test_items))
            for rate in rates:
                for arm in arms:
                    source_arm = ARM_SOURCES[arm]
                    cell_path = source_cell_path(summary_path, source_arm, rate, seed)
                    source_cell = load_json(cell_path)
                    source_sha = sha256_file(cell_path)
                    base_package, merge_info, tool_doc_hash = rebuild_source_package(packages_root, seed, source_arm, rate, progress=progress)
                    expected_hash = str(source_cell["compendium"]["global_sha256"])
                    actual_hash = hash_package(base_package)
                    if merge_info["global_sha256"] != expected_hash or actual_hash != expected_hash:
                        raise RuntimeError(
                            f"Source hash mismatch for {arm} seed={seed} rate={rate}: expected {expected_hash}, "
                            f"merge_info={merge_info['global_sha256']} rebuilt={actual_hash}"
                        )
                    decode_failures = 0
                    if arm == "typed_merge_flat_render":
                        eval_package = relabel_as_flat(base_package)
                    elif arm == "flat_merge_structured_render":
                        eval_package, decode_failures = decode_flat_payload(base_package)
                    else:
                        eval_package = base_package
                    progress.log("arm_begin", seed=seed, rate=rate, arm=arm, source_cell=str(cell_path))
                    result = evaluate_reranker_arm(
                        arm,
                        eval_package,
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
                        "arm": arm,
                        "render_mode": ARM_RENDER_MODE[arm],
                        "merge_mode": ARM_MERGE_MODE[arm],
                        "source_arm": source_arm,
                        "source_cell_path": str(cell_path),
                        "source_cell_sha256": source_sha,
                        "source_accuracy": float(source_cell.get("accuracy", 0.0)),
                        "source_recall_at_5": float(source_cell.get("recall_at_5", 0.0)),
                        "baseline_replay_check": not args.skip_baseline_replay_check,
                        "source_compendium_sha256": expected_hash,
                        "replayed_compendium_sha256": hash_package(eval_package),
                        "tool_doc_sha256": tool_doc_hash,
                        "decode_failure_count": decode_failures,
                        "embedder": embedder,
                        "git_commit": commit,
                        "retrieval_mode": args.retrieval_mode,
                        "retrieval_pool_size": args.retrieval_pool_size,
                        "reranker_variant": args.reranker_variant,
                        "paper_eligible": True,
                        "metric_definition": {
                            "correct": "predicted_tool in gold_parent_tools",
                            "recall_at_5": "any gold tool present among candidate tools",
                        },
                    })
                    if arm in {"typed_conflictlog", "flat_majority"} and not args.skip_baseline_replay_check:
                        if f"{result['accuracy']:.3f}" != f"{float(source_cell['accuracy']):.3f}" or f"{result['recall_at_5']:.3f}" != f"{float(source_cell['recall_at_5']):.3f}":
                            raise RuntimeError(
                                f"Baseline replay mismatch for {arm} seed={seed} rate={rate}: "
                                f"replay=({result['accuracy']:.3f}, {result['recall_at_5']:.3f}) "
                                f"source=({float(source_cell['accuracy']):.3f}, {float(source_cell['recall_at_5']):.3f})"
                            )
                    out_path = args.output_dir / arm / f"conflict_{rate:02d}" / f"seed_{seed}.json"
                    save_json(out_path, result)
                    decode_failures_by_cell[f"{arm}:{seed}:{rate}"] = decode_failures
                    all_results.append(result)
                    progress.log("arm_done", seed=seed, rate=rate, arm=arm, accuracy=result["accuracy"], recall_at_5=result["recall_at_5"])
        summary = {
            "paper_eligible": True,
            "config": {
                "git_commit": commit,
                "seeds": seeds,
                "rates": rates,
                "arms": arms,
                "retrieval_mode": args.retrieval_mode,
                "retrieval_pool_size": args.retrieval_pool_size,
                "top_k": args.top_k,
                "reranker_variant": args.reranker_variant,
                "embed_model": args.embed_model,
                "model_path": args.model_path,
                "local_embedder": embedder,
            },
            "per_arm": aggregate(all_results, arms, rates),
            "decode_failures_by_cell": decode_failures_by_cell,
        }
        save_json(args.output_dir / "summary.json", summary)
        progress.log("complete", output_dir=str(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
