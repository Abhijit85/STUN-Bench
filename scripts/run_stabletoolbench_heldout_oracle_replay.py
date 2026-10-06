#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_ROOT = Path(os.environ.get("FEDRAG_CANONICAL_ROOT", "<REPO_ROOT>")).resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from math_qa import JinaAIClient
from scripts.run_gsm8k_small_router_sweep import parse_seed_list
from scripts.run_stabletoolbench_federated import (
    DEFAULT_MODEL_PATH,
    DEFAULT_STB_ROOT,
    ProgressLogger,
    assert_clean_tree,
    load_local_backend,
    load_package_file,
    resolve_local_embedder,
    save_json,
    temporary_env,
)
from scripts.run_stabletoolbench_heldout import (
    HELDOUT_GROUPS,
    evaluate_oracle_arm,
    heldout_tool_set,
    parse_csv,
)
from scripts.run_stabletoolbench_typing_isolation import load_eval_queries


DEFAULT_OUTPUT_DIR = CANONICAL_ROOT / "artifacts" / "results" / "stabletoolbench_heldout_oracle_distinct5_r1"

PACKAGE_ROOTS = {
    42: CANONICAL_ROOT / "artifacts" / "results" / "stabletoolbench_heldout_retriever_compare_e6_r4" / "seed_42" / "packages",
    123: CANONICAL_ROOT / "artifacts" / "results" / "stabletoolbench_heldout_retriever_compare_e6_seed123_r4" / "seed_123" / "packages",
    456: CANONICAL_ROOT / "artifacts" / "results" / "stabletoolbench_heldout_retriever_compare_e6_seed456_r4" / "seed_456" / "packages",
}

REFERENCE_ROOTS = {
    42: CANONICAL_ROOT / "artifacts" / "results" / "stabletoolbench_heldout_distinct5_d1b_r9" / "seed_42",
    123: CANONICAL_ROOT / "artifacts" / "results" / "stabletoolbench_heldout_distinct5_d1b_seed123_r1" / "seed_123",
    456: CANONICAL_ROOT / "artifacts" / "results" / "stabletoolbench_heldout_distinct5_d1b_seed456_r1" / "seed_456",
}

FLAT_REFERENCE_ROOTS = {
    42: REFERENCE_ROOTS[42],
    123: CANONICAL_ROOT / "artifacts" / "results" / "stabletoolbench_heldout_distinct5_d1b_seed123_flatonly_r1" / "seed_123",
    456: REFERENCE_ROOTS[456],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay held-out oracle reranking from persisted held-out packages.")
    parser.add_argument("--seeds", type=str, default="42,123,456")
    parser.add_argument("--arms", type=str, default="synapse,flat_pool")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retrieval-pool-size", type=int, default=200)
    parser.add_argument("--retrieval-mode", type=str, default="distinct_tool_topk")
    parser.add_argument("--reranker-variant", type=str, default="V3")
    parser.add_argument("--embed-model", type=str, default="jina-embeddings-v2-base-en")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--stb-root", type=Path, default=DEFAULT_STB_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values) if values else 0.0,
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def package_path(seed: int, arm: str) -> Path:
    filename = "synapse_shared.json" if arm == "synapse" else "flat_pool.json"
    return PACKAGE_ROOTS[seed] / filename


def reference_path(seed: int, arm: str) -> Path:
    root = FLAT_REFERENCE_ROOTS[seed] if arm == "flat_pool" else REFERENCE_ROOTS[seed]
    return root / f"{arm}.json"


def main() -> int:
    load_dotenv(REPO_ROOT / ".env")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    commit, dirty_entries = assert_clean_tree(allow_dirty=args.allow_dirty)
    progress = ProgressLogger(args.output_dir)
    seeds = parse_seed_list(args.seeds)
    arms = parse_csv(args.arms)
    progress.log("start", git_commit=commit, dirty_entries=dirty_entries, seeds=seeds, arms=arms)

    embedder = resolve_local_embedder()
    jina_client = JinaAIClient(api_keys=[])
    backend = load_local_backend(args.model_path)

    # The E6 package roots include the docs package needed by the shared and flat packages.
    first_tool_doc, _tool_doc_hash = load_package_file(PACKAGE_ROOTS[seeds[0]] / "tool_docs.json")
    queries = load_eval_queries(
        args.stb_root,
        ["G1_instruction", "G1_tool", "G1_category", "G2_instruction", "G2_category", "G3_instruction"],
        first_tool_doc,
    )
    heldout_tools, heldout_sanity = heldout_tool_set(queries, HELDOUT_GROUPS)
    heldout_set = set(heldout_tools)
    progress.log("queries_loaded", query_count=len(queries), heldout_count=sum(1 for q in queries if any(t in heldout_set for t in q.gold_tools)), heldout_sanity=heldout_sanity)

    all_results: dict[str, list[dict[str, Any]]] = {arm: [] for arm in arms}
    with temporary_env(
        {
            "JINA_LOCAL_EMBED_MODEL": embedder["model_path"],
            "JINA_LOCAL_EMBED_DEVICE": embedder["device"],
            "JINA_LOCAL_EMBED_LOCAL_ONLY": embedder["local_only"],
            "JINA_API_KEY": None,
        }
    ):
        for seed in seeds:
            seed_dir = args.output_dir / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            for arm in arms:
                pkg_path = package_path(seed, arm)
                ref_path = reference_path(seed, arm)
                package, package_hash = load_package_file(pkg_path)
                ref = load_json(ref_path)
                ref_hash = ref.get("compendium", {}).get("global_sha256")
                ref_match = package_hash == ref_hash
                progress.log(
                    "arm_loaded",
                    seed=seed,
                    arm=arm,
                    package_path=str(pkg_path),
                    package_sha256=package_hash,
                    reference_path=str(ref_path),
                    reference_compendium_sha256=ref_hash,
                    reference_hash_match=ref_match,
                    artifact_count=len(package.artifacts),
                )
                result = evaluate_oracle_arm(
                    arm,
                    package,
                    queries,
                    heldout_set,
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
                result.update(
                    {
                        "paper_eligible": True,
                        "repo_commit": commit,
                        "seed": seed,
                        "arm": f"{arm}_oracle",
                        "source_package_path": str(pkg_path),
                        "source_package_sha256": package_hash,
                        "reference_d1b_path": str(ref_path),
                        "reference_d1b_compendium_sha256": ref_hash,
                        "reference_d1b_hash_match": ref_match,
                        "candidate_rule": "distinct5_walkdown",
                        "oracle_insertion_rule": "each missing gold tool replaces the lowest-ranked non-gold candidate, preserving k=5",
                        "oracle_distractor_source": "the four non-gold candidates are drawn from this arm's own index before oracle replacement",
                        "retrieval_pool_size": args.retrieval_pool_size,
                        "top_k": args.top_k,
                        "retrieval_mode": args.retrieval_mode,
                        "reranker_variant": args.reranker_variant,
                        "embedder": embedder,
                        "metric_definition": {
                            "correct": "predicted_tool in gold_parent_tools",
                            "heldout": "all gold tools are in H",
                        },
                    }
                )
                save_json(seed_dir / f"{arm}_oracle.json", result)
                all_results[arm].append(result)
                progress.log(
                    "arm_done",
                    seed=seed,
                    arm=arm,
                    accuracy=result["accuracy"],
                    oracle_insertion_fraction=result["oracle_insertion_fraction"],
                    heldout_accuracy=result["subset_metrics"]["heldout"]["accuracy"],
                )

    summary = {
        "paper_eligible": True,
        "config": {
            "repo_commit": commit,
            "seeds": seeds,
            "arms": arms,
            "candidate_rule": "distinct5_walkdown",
            "oracle_insertion_rule": "replace-lowest-ranked-non-gold-preserve-k5",
            "oracle_distractor_source": "arm_own_index",
            "retrieval_mode": args.retrieval_mode,
            "retrieval_pool_size": args.retrieval_pool_size,
            "top_k": args.top_k,
            "reranker_variant": args.reranker_variant,
            "embed_model": args.embed_model,
            "model_path": args.model_path,
            "local_embedder": embedder,
        },
        "per_arm": {},
    }
    for arm, results in all_results.items():
        acc = [r["accuracy"] for r in results]
        ins = [r["oracle_insertion_fraction"] for r in results]
        summary["per_arm"][f"{arm}_oracle"] = {
            "accuracy": summarize(acc),
            "oracle_insertion_fraction": summarize(ins),
            "seeds": [r["seed"] for r in results],
            "heldout_by_group": {
                group: {
                    "accuracy": summarize([r["subset_metrics"]["heldout_by_group"][group]["accuracy"] for r in results]),
                    "recall_at_5": summarize([r["subset_metrics"]["heldout_by_group"][group]["recall_at_5"] for r in results]),
                    "n": [r["subset_metrics"]["heldout_by_group"][group]["n"] for r in results],
                }
                for group in sorted(results[0]["subset_metrics"]["heldout_by_group"])
            },
            "source_packages": [
                {
                    "seed": r["seed"],
                    "path": r["source_package_path"],
                    "sha256": r["source_package_sha256"],
                    "reference_d1b_hash_match": r["reference_d1b_hash_match"],
                }
                for r in results
            ],
        }
    save_json(args.output_dir / "summary.json", summary)
    progress.log("complete", output_dir=str(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
