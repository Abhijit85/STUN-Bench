#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from math_qa import JinaAIClient
from scripts.run_stabletoolbench_federated import (
    GROUPS,
    DEFAULT_STB_ROOT,
    DEFAULT_TOOLBENCH_INSTRUCTION_DIR,
    ProgressLogger,
    aggregate_seed_summaries,
    apply_junk_filter,
    assert_clean_tree,
    assert_no_eval_overlap,
    assign_clients,
    build_tool_registry,
    evaluate_classifier_arm,
    filter_eval_queries,
    filter_experience_items,
    limit_client_items,
    limit_experience_items,
    load_stabletoolbench_queries,
    load_toolbench_training_items,
    remove_exact_eval_overlaps,
    remove_near_duplicate_eval_overlaps,
    resolve_local_embedder,
    save_json,
    stable_hash,
    temporary_env,
    train_query_classifier,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean StableToolBench query-classifier rerun on the leak-filtered per-seed client pool.")
    parser.add_argument("--stb-root", type=Path, default=DEFAULT_STB_ROOT)
    parser.add_argument("--toolbench-instruction-dir", type=Path, default=DEFAULT_TOOLBENCH_INSTRUCTION_DIR)
    parser.add_argument("--groups", type=str, default=",".join(GROUPS))
    parser.add_argument("--seeds", type=str, default="42,123,456")
    parser.add_argument("--client-count", type=int, default=5)
    parser.add_argument("--partition-mode", type=str, default="category", choices=["category", "iid"])
    parser.add_argument("--max-train-items", type=int, default=0)
    parser.add_argument("--max-items-per-client", type=int, default=5000)
    parser.add_argument("--junk-filter", action="store_true")
    parser.add_argument("--embed-model", type=str, default="jina-embeddings-v2-base-en")
    parser.add_argument("--pool-embedding-cache-dir", type=Path, required=True)
    parser.add_argument("--contamination-near-duplicate-threshold", type=float, default=0.95)
    parser.add_argument("--contamination-report-threshold", type=float, default=0.90)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args()


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_seeds(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def filtered_pool_sha(items: list[object]) -> str:
    return stable_hash(
        [
            {
                "query_id": item.query_id,
                "query": item.query,
                "group": item.group,
                "gold_tools": list(item.gold_tools),
                "primary_category": item.primary_category,
            }
            for item in items
        ]
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress = ProgressLogger(args.output_dir)
    commit, dirty_entries = assert_clean_tree(allow_dirty=args.allow_dirty)

    groups = parse_csv(args.groups)
    seeds = parse_seeds(args.seeds)

    progress.log("load_queries_begin", groups=groups)
    queries = load_stabletoolbench_queries(args.stb_root, groups)
    progress.log("load_queries_done", query_count=len(queries))

    progress.log("load_training_begin", instruction_dir=str(args.toolbench_instruction_dir))
    train_items = load_toolbench_training_items(args.toolbench_instruction_dir)
    progress.log("load_training_done", train_count=len(train_items))

    junk_filter_info = {
        "enabled": False,
        "dropped_tool_count": 0,
        "dropped_query_count": 0,
        "dropped_train_item_count": 0,
    }
    if args.junk_filter:
        registry = build_tool_registry(queries + train_items)
        registry, registry_filter_info = apply_junk_filter(registry)
        queries, query_filter_info = filter_eval_queries(queries, registry)
        train_items, train_filter_info = filter_experience_items(train_items, registry)
        junk_filter_info = {**registry_filter_info, **query_filter_info, **train_filter_info}
        progress.log("junk_filter_done", query_count=len(queries), train_count=len(train_items), **junk_filter_info)

    if args.max_train_items > 0:
        original_train_count = len(train_items)
        train_items = limit_experience_items(train_items, args.max_train_items, seeds[0] if seeds else 0)
        progress.log(
            "pre_filter_train_cap_applied",
            original_train_count=original_train_count,
            capped_train_count=len(train_items),
            max_train_items=args.max_train_items,
        )

    embedder_info = resolve_local_embedder()
    jina_client = JinaAIClient(api_keys=[])
    progress.log(
        "init_embedding_client_done",
        embed_model=args.embed_model,
        local_model=embedder_info["model_path"],
        embedder_revision=embedder_info["revision"],
        embedder_device=embedder_info["device"],
    )

    contamination_filter_info = {
        "enabled": True,
        "near_duplicate_threshold": args.contamination_near_duplicate_threshold,
        "report_threshold": args.contamination_report_threshold,
    }

    with temporary_env(
        {
            "JINA_LOCAL_EMBED_MODEL": embedder_info["model_path"],
            "JINA_LOCAL_EMBED_DEVICE": embedder_info["device"],
            "JINA_LOCAL_EMBED_LOCAL_ONLY": embedder_info["local_only"],
            "JINA_API_KEY": None,
        }
    ):
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
            int(contamination_filter_info["pool_items_removed_exact"]) + int(contamination_filter_info["pool_items_removed_near_dup"])
        )
        contamination_filter_info["eval_queries_removed"] = 0
        post_filter_refusal = assert_no_eval_overlap(
            train_items,
            queries,
            forbidden_near_duplicate_texts=forbidden_near_duplicate_texts,
            stage="post_filter_training_pool",
        )
        contamination_filter_info["post_filter_refusal_check"] = post_filter_refusal
        contamination_filter_info["filtered_pool_path"] = str(args.output_dir / "filtered_pool.json")
        contamination_filter_info["filtered_pool_sha256"] = filtered_pool_sha(train_items)
        contamination_filter_info["filtered_pool_count"] = len(train_items)
        progress.log("contamination_filter_done", train_count=len(train_items), test_count=len(queries), **contamination_filter_info)

        save_json(
            args.output_dir / "filtered_pool.json",
            {
                "repo_commit": commit,
                "groups": groups,
                "query_count": len(queries),
                "filtered_pool_count": len(train_items),
                "filtered_pool_sha256": contamination_filter_info["filtered_pool_sha256"],
                "contamination_filter": contamination_filter_info,
                "items": [
                    {
                        "query_id": item.query_id,
                        "query": item.query,
                        "group": item.group,
                        "gold_tools": list(item.gold_tools),
                        "primary_category": item.primary_category,
                    }
                    for item in train_items
                ],
            },
        )

        results = []
        for seed in seeds:
            progress.log("seed_begin", seed=seed)
            seed_train_items = list(train_items)
            clients = assign_clients(seed_train_items, args.client_count, args.partition_mode, seed)
            if args.max_items_per_client > 0:
                original_client_sizes = {client_id: len(items) for client_id, items in sorted(clients.items())}
                clients = limit_client_items(clients, args.max_items_per_client, seed)
                progress.log(
                    "client_cap_applied",
                    seed=seed,
                    original_client_sizes=original_client_sizes,
                    capped_client_sizes={client_id: len(items) for client_id, items in sorted(clients.items())},
                    max_items_per_client=args.max_items_per_client,
                )
            classifier_fit_items = [item for client_id in sorted(clients) for item in clients[client_id]]
            classifier_fit_refusal = assert_no_eval_overlap(
                classifier_fit_items,
                queries,
                forbidden_near_duplicate_texts=forbidden_near_duplicate_texts,
                stage=f"seed_{seed}_classifier_fit_set",
            )
            fit_pool_sha = filtered_pool_sha(classifier_fit_items)
            progress.log(
                "classifier_fit_ready",
                seed=seed,
                classifier_train_n=len(classifier_fit_items),
                filtered_pool_path=str(args.output_dir / "filtered_pool.json"),
                filtered_pool_sha256=contamination_filter_info["filtered_pool_sha256"],
                classifier_fit_pool_sha256=fit_pool_sha,
                classifier_fit_refusal_check=classifier_fit_refusal,
            )
            bundle = train_query_classifier(classifier_fit_items)
            result = evaluate_classifier_arm(bundle, queries)
            result.update(
                {
                    "seed": seed,
                    "paper_eligible": True,
                    "repo_commit": commit,
                    "data_mode": "toolbench_train",
                    "embedder": embedder_info,
                    "junk_filter": junk_filter_info,
                    "contamination_filter": contamination_filter_info,
                    "classifier_train_n": len(classifier_fit_items),
                    "filtered_pool_path": str(args.output_dir / "filtered_pool.json"),
                    "filtered_pool_sha256": contamination_filter_info["filtered_pool_sha256"],
                    "classifier_fit_pool_sha256": fit_pool_sha,
                    "classifier_fit_refusal_check": classifier_fit_refusal,
                    "client_sizes": {client_id: len(items) for client_id, items in sorted(clients.items())},
                    "allow_dirty": args.allow_dirty,
                    "dirty_entry_count": len(dirty_entries),
                }
            )
            save_json(args.output_dir / f"seed_{seed}.json", result)
            results.append(result)
            progress.log("arm_done", seed=seed, arm="query_classifier_clean", accuracy=result["accuracy"], recall_at_5=result["recall_at_5"])

        combined = {
            "paper_eligible": True,
            "config": {
                "groups": groups,
                "seeds": seeds,
                "data_mode": "toolbench_train",
                "partition_mode": args.partition_mode,
                "client_count": args.client_count,
                "max_train_items": args.max_train_items,
                "max_items_per_client": args.max_items_per_client,
                "junk_filter": args.junk_filter,
                "embed_model": args.embed_model,
                "local_embedder": embedder_info,
                "git_commit": commit,
                "allow_dirty": args.allow_dirty,
            },
            "query_classifier": aggregate_seed_summaries(results),
            "contamination_filter": contamination_filter_info,
        }
        save_json(args.output_dir / "combined_summary.json", combined)
        progress.log(
            "complete",
            seed_count=len(seeds),
            overall_accuracy=combined["query_classifier"]["mean_accuracy"],
            overall_sd=combined["query_classifier"]["sd_accuracy"],
        )

        group_metrics = combined["query_classifier"]["group_metrics"]
        print(f"overall_accuracy={combined['query_classifier']['mean_accuracy']:.6f} sd={combined['query_classifier']['sd_accuracy']:.6f}")
        for group in groups:
            metrics = group_metrics[group]
            print(f"{group} n={metrics['count']} acc={metrics['mean_accuracy']:.6f} sd={metrics['sd_accuracy']:.6f}")


if __name__ == "__main__":
    main()
