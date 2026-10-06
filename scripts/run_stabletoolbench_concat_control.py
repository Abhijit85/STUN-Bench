#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_ROOT = Path(os.environ.get("FEDRAG_CANONICAL_ROOT", str(REPO_ROOT))).resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from math_qa import JinaAIClient
from scripts.run_stabletoolbench_federated import (
    DEFAULT_MODEL_PATH,
    DEFAULT_STB_ROOT,
    GROUPS,
    ProgressLogger,
    assert_clean_tree,
    combine_packages,
    evaluate_reranker_arm,
    hash_package,
    load_local_backend,
    maybe_cuda_synchronize,
    resolve_local_embedder,
    save_json,
    temporary_env,
)
from scripts.run_stabletoolbench_typing_isolation import inject_contradictions, load_eval_queries, load_seed_packages

DEFAULT_PACKAGE_ROOTS = {
    42: CANONICAL_ROOT / "artifacts" / "verification" / "stabletoolbench_clean_anchor_seed42_r4" / "packages",
    123: CANONICAL_ROOT / "artifacts" / "verification" / "stabletoolbench_clean_anchor_seed123_r4" / "packages",
    456: CANONICAL_ROOT / "artifacts" / "verification" / "stabletoolbench_clean_anchor_seed456_r4" / "packages",
}
DEFAULT_OUTPUT_DIR = CANONICAL_ROOT / "artifacts" / "results" / "stabletoolbench_concat_control_r1"
SUBSET_GROUPS = ["G1_instruction", "G2_instruction"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="StableToolBench FICAL-style concatenation control.")
    parser.add_argument("--seeds", type=str, default="42,123,456")
    parser.add_argument("--rates", type=str, default="0,60")
    parser.add_argument("--full-rates", type=str, default="0")
    parser.add_argument("--packages-root", action="append", default=[])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retrieval-pool-size", type=int, default=20)
    parser.add_argument("--retrieval-mode", type=str, default="distinct_tool_topk")
    parser.add_argument("--reranker-variant", type=str, default="V3")
    parser.add_argument("--embed-model", type=str, default="jina-embeddings-v2-base-en")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--stb-root", type=Path, default=DEFAULT_STB_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args()


def parse_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def resolve_roots(args: argparse.Namespace, seeds: list[int]) -> dict[int, Path]:
    if not args.packages_root:
        return {seed: DEFAULT_PACKAGE_ROOTS[seed] for seed in seeds}
    roots = [Path(item) for item in args.packages_root]
    if len(roots) == 1:
        return {seed: roots[0] for seed in seeds}
    if len(roots) != len(seeds):
        raise ValueError("--packages-root must be omitted, passed once, or passed once per seed")
    return {seed: root for seed, root in zip(seeds, roots)}


def concat_package(tool_doc_package, client_packages: dict[str, Any]):
    artifacts = list(tool_doc_package.artifacts)
    for client_id, package in sorted(client_packages.items()):
        artifacts.extend(copy.deepcopy(package.artifacts))
    package, package_hash = combine_packages("concat_with_docs", [tool_doc_package, *[client_packages[k] for k in sorted(client_packages)]])
    # combine_packages preserves all artifacts while producing the same hash convention used elsewhere.
    return package, package_hash


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for scope in sorted({r["scope"] for r in results}):
        out[scope] = {}
        for rate in sorted({r["rate"] for r in results if r["scope"] == scope}):
            subset = [r for r in results if r["scope"] == scope and r["rate"] == rate]
            acc = [float(r["accuracy"]) for r in subset]
            rec = [float(r["recall_at_5"]) for r in subset]
            out[scope][str(rate)] = {
                "mean_accuracy": statistics.mean(acc) if acc else 0.0,
                "sd_accuracy": statistics.stdev(acc) if len(acc) > 1 else 0.0,
                "mean_recall_at_5": statistics.mean(rec) if rec else 0.0,
                "sd_recall_at_5": statistics.stdev(rec) if len(rec) > 1 else 0.0,
                "per_seed": {str(r["seed"]): {"accuracy": r["accuracy"], "recall_at_5": r["recall_at_5"], "n": len(r.get("rows", []))} for r in subset},
            }
    return out


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    commit, dirty = assert_clean_tree(allow_dirty=args.allow_dirty)
    progress = ProgressLogger(args.output_dir)
    progress.log("launch", git_commit=commit, dirty_entry_count=len(dirty))
    seeds = parse_ints(args.seeds)
    rates = parse_ints(args.rates)
    full_rates = parse_ints(args.full_rates)
    roots = resolve_roots(args, seeds)
    embedder = resolve_local_embedder()
    jina_client = JinaAIClient(api_keys=[])
    results: list[dict[str, Any]] = []
    with temporary_env({
        "JINA_LOCAL_EMBED_MODEL": embedder["model_path"],
        "JINA_LOCAL_EMBED_DEVICE": embedder["device"],
        "JINA_LOCAL_EMBED_LOCAL_ONLY": embedder["local_only"],
        "JINA_API_KEY": None,
    }):
        backend = load_local_backend(args.model_path)
        for seed in seeds:
            root = roots[seed]
            client_packages, tool_doc_package, tool_doc_hash = load_seed_packages(root, seed)
            eval_sets = {
                "subset_g1g2_instruction": load_eval_queries(args.stb_root, SUBSET_GROUPS, tool_doc_package),
                "full": load_eval_queries(args.stb_root, GROUPS, tool_doc_package),
            }
            progress.log("seed_inputs_ready", seed=seed, packages_root=str(root), client_count=len(client_packages), tool_doc_sha256=tool_doc_hash, subset_n=len(eval_sets["subset_g1g2_instruction"]), full_n=len(eval_sets["full"]))
            for scope, test_items in eval_sets.items():
                scope_rates = rates if scope == "subset_g1g2_instruction" else full_rates
                for rate in scope_rates:
                    mutated, contradiction_info = inject_contradictions(client_packages, rate, seed)
                    package, package_hash = concat_package(tool_doc_package, mutated)
                    progress.log("cell_begin", seed=seed, scope=scope, rate=rate, contradiction_hash=contradiction_info["contradiction_hash"], package_sha256=package_hash, artifact_count=len(package.artifacts), query_count=len(test_items))
                    result = evaluate_reranker_arm("concat_control", package, test_items, jina_client, args.embed_model, backend, args.top_k, args.retrieval_pool_size, args.retrieval_mode, args.reranker_variant, progress=progress, seed=seed)
                    result.update({
                        "paper_eligible": True,
                        "git_commit": commit,
                        "seed": seed,
                        "scope": scope,
                        "rate": rate,
                        "arm": "concat_control",
                        "merge_mode": "server_concatenate_no_dedup_no_field_merge",
                        "render_mode": "structured_description_first",
                        "compendium": {"global_sha256": package_hash, "artifact_count": len(package.artifacts)},
                        "tool_doc_sha256": tool_doc_hash,
                        "contradiction_hash": contradiction_info["contradiction_hash"],
                        "contradicted_count": contradiction_info["contradicted_count"],
                        "contradicted_signatures": contradiction_info["contradicted_signatures"],
                        "embedder": embedder,
                        "retrieval_mode": args.retrieval_mode,
                        "retrieval_pool_size": args.retrieval_pool_size,
                        "top_k": args.top_k,
                        "reranker_variant": args.reranker_variant,
                    })
                    out = args.output_dir / scope / f"conflict_{rate:02d}" / f"seed_{seed}.json"
                    save_json(out, result)
                    results.append(result)
                    progress.log("cell_done", seed=seed, scope=scope, rate=rate, accuracy=result["accuracy"], recall_at_5=result["recall_at_5"], output=str(out))
    summary = {
        "paper_eligible": True,
        "config": {
            "git_commit": commit,
            "seeds": seeds,
            "rates": rates,
            "full_rates": full_rates,
            "top_k": args.top_k,
            "retrieval_pool_size": args.retrieval_pool_size,
            "retrieval_mode": args.retrieval_mode,
            "reranker_variant": args.reranker_variant,
            "model_path": args.model_path,
            "local_embedder": embedder,
            "packages_roots": {str(seed): str(roots[seed]) for seed in seeds},
        },
        "aggregate": aggregate(results),
    }
    save_json(args.output_dir / "summary.json", summary)
    progress.log("complete", output_dir=str(args.output_dir))
    print(json.dumps(summary["aggregate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
