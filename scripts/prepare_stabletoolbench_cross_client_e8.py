#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_gsm8k_small_router_sweep import parse_seed_list
from scripts.run_stabletoolbench_federated import (
    DEFAULT_STB_ROOT,
    DEFAULT_TOOLBENCH_INSTRUCTION_DIR,
    GROUPS,
    ProgressLogger,
    apply_junk_filter,
    assert_clean_tree,
    assign_clients,
    build_tool_registry,
    filter_eval_queries,
    filter_experience_items,
    limit_client_items,
    load_stabletoolbench_queries,
    load_toolbench_training_items,
    stable_hash,
)
from scripts.run_stabletoolbench_heldout import (
    HELDOUT_GROUPS,
    filter_heldout_pool,
    heldout_tool_set,
    sha256_texts,
)


DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "results" / "stabletoolbench_cross_client_e8_inputs_r1"
TX_GROUP = "G1_instruction"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="E8 step 1: prepare cross-client ownership assignments and forced client draws."
    )
    p.add_argument("--stb-root", type=Path, default=DEFAULT_STB_ROOT)
    p.add_argument("--toolbench-instruction-dir", type=Path, default=DEFAULT_TOOLBENCH_INSTRUCTION_DIR)
    p.add_argument("--groups", default=",".join(GROUPS))
    p.add_argument("--seeds", default="42,123,456")
    p.add_argument("--client-count", type=int, default=5)
    p.add_argument("--max-items-per-client", type=int, default=5000)
    p.add_argument("--partition-mode", choices=("category", "iid"), default="category")
    p.add_argument("--target-items-per-tool", type=int, default=8)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--allow-dirty", action="store_true")
    return p.parse_args()


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def owner_for_tool(tool: str, seed: int, client_count: int) -> str:
    digest = hashlib.sha256(json.dumps({"e8_owner": seed, "tool": tool}, sort_keys=True).encode()).hexdigest()
    return f"client_{int(digest[:12], 16) % client_count}"


def item_has_tool(item: Any, tools: set[str]) -> bool:
    return any(tool in tools for tool in item.gold_tools)


def item_key(item: Any) -> str:
    return str(item.query_id)


def item_record(item: Any) -> dict[str, Any]:
    return {
        "query_id": item.query_id,
        "query": item.query,
        "gold_tools": list(item.gold_tools),
        "group": item.group,
        "primary_category": item.primary_category,
        "categories": list(item.categories),
    }


def force_single_owner_draws(
    base_clients: dict[str, list[Any]],
    train_items: list[Any],
    tx_tools: set[str],
    owners: dict[str, str],
    *,
    seed: int,
    max_items_per_client: int,
    target_items_per_tool: int,
) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    rng = random.Random(stable_hash({"e8_force": seed}))
    clients = {client_id: [item for item in items if not item_has_tool(item, tx_tools)] for client_id, items in base_clients.items()}
    by_tool: dict[str, list[Any]] = defaultdict(list)
    for item in train_items:
        tx_labels = sorted(set(item.gold_tools) & tx_tools)
        if len(tx_labels) == 1:
            by_tool[tx_labels[0]].append(item)

    injected: dict[str, int] = {}
    missing_tools: list[str] = []
    evicted: dict[str, int] = {client_id: 0 for client_id in clients}
    for tool in sorted(tx_tools):
        candidates = list(by_tool.get(tool, []))
        if not candidates:
            missing_tools.append(tool)
            continue
        rng.shuffle(candidates)
        owner = owners[tool]
        existing_ids = {item_key(item) for item in clients[owner]}
        chosen = [item for item in candidates if item_key(item) not in existing_ids][:target_items_per_tool]
        clients[owner].extend(chosen)
        injected[tool] = len(chosen)
        while len(clients[owner]) > max_items_per_client:
            # Evict a non-Tx item first, preserving the forced ownership examples.
            idx = next((i for i, item in enumerate(clients[owner]) if not item_has_tool(item, tx_tools)), None)
            if idx is None:
                raise RuntimeError(f"client {owner} exceeds cap using only forced Tx items")
            clients[owner].pop(idx)
            evicted[owner] += 1

    coverage: dict[str, list[str]] = {tool: [] for tool in tx_tools}
    mention_counts: dict[str, int] = {tool: 0 for tool in tx_tools}
    for client_id, items in clients.items():
        for tool in tx_tools:
            label_count = sum(1 for item in items if tool in item.gold_tools)
            if label_count:
                coverage[tool].append(client_id)
            # Mentions are reported, not removed: this is the intended E8 distinction.
            needle = tool.replace("_", " ").replace("-", " ").lower()
            mention_counts[tool] += sum(1 for item in items if needle in item.query.lower() and tool not in item.gold_tools)

    bad = {tool: clients_for_tool for tool, clients_for_tool in coverage.items() if clients_for_tool != [owners[tool]]}
    if bad:
        raise RuntimeError(f"exactly-one-owner assertion failed for {len(bad)} Tx tools: {dict(list(bad.items())[:5])}")

    return clients, {
        "target_items_per_tool": target_items_per_tool,
        "missing_tx_tools": missing_tools,
        "injected_items_per_tool": injected,
        "evicted_non_tx_items_per_client": evicted,
        "non_owner_mention_count_total": sum(mention_counts.values()),
        "non_owner_mention_count_by_tool_sample": dict(list(sorted(mention_counts.items()))[:20]),
        "owner_coverage": coverage,
    }


def main() -> int:
    load_dotenv(REPO_ROOT / ".env")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    commit, dirty = assert_clean_tree(allow_dirty=args.allow_dirty)
    progress = ProgressLogger(args.output_dir)
    progress.log("start", git_commit=commit, dirty_entry_count=len(dirty))

    groups = parse_csv(args.groups)
    seeds = parse_seed_list(args.seeds)
    queries = load_stabletoolbench_queries(args.stb_root, groups)
    train_items = load_toolbench_training_items(args.toolbench_instruction_dir)
    registry = build_tool_registry(queries + train_items)
    registry, registry_filter = apply_junk_filter(registry)
    queries, query_filter = filter_eval_queries(queries, registry)
    train_items, train_filter = filter_experience_items(train_items, registry)
    progress.log(
        "data_loaded",
        query_count=len(queries),
        train_count=len(train_items),
        registry_tool_count=len(registry),
        **registry_filter,
        **query_filter,
        **train_filter,
    )

    heldout_tools, heldout_sanity = heldout_tool_set(queries, HELDOUT_GROUPS)
    heldout_set = set(heldout_tools)
    train_items, heldout_filter = filter_heldout_pool(train_items, heldout_set)
    if heldout_filter["remaining_items_with_heldout_label"] != 0:
        raise RuntimeError("held-out H labels leaked into E8 source pool")
    heldout_sha = sha256_texts(heldout_tools)

    requested_tx_tools = sorted({tool for query in queries if query.group == TX_GROUP for tool in query.gold_tools} - heldout_set)
    requested_tx_set = set(requested_tx_tools)
    safe_tx_tools = {
        next(iter(set(item.gold_tools) & requested_tx_set))
        for item in train_items
        if len(set(item.gold_tools) & requested_tx_set) == 1
    }
    tx_tools = sorted(requested_tx_set & safe_tx_tools)
    excluded_tx_tools = sorted(requested_tx_set - set(tx_tools))
    if not tx_tools:
        raise RuntimeError("no Tx tools found")
    tx_query_ids = sorted(query.query_id for query in queries if query.group == TX_GROUP and any(tool in tx_tools for tool in query.gold_tools))
    progress.log(
        "split_ready",
        heldout_tool_count=len(heldout_tools),
        heldout_tools_sha256=heldout_sha,
        requested_tx_tool_count=len(requested_tx_tools),
        tx_tool_count=len(tx_tools),
        excluded_tx_tool_count=len(excluded_tx_tools),
        tx_query_count=len(tx_query_ids),
    )

    per_seed: dict[str, Any] = {}
    for seed in seeds:
        owners = {tool: owner_for_tool(tool, seed, args.client_count) for tool in tx_tools}
        base_clients = assign_clients(train_items, args.client_count, args.partition_mode, seed)
        capped_clients = limit_client_items(base_clients, args.max_items_per_client, seed)
        clients, force_info = force_single_owner_draws(
            capped_clients,
            train_items,
            set(tx_tools),
            owners,
            seed=seed,
            max_items_per_client=args.max_items_per_client,
            target_items_per_tool=args.target_items_per_tool,
        )
        client_sizes = {client_id: len(items) for client_id, items in sorted(clients.items())}
        if any(size > args.max_items_per_client for size in client_sizes.values()):
            raise RuntimeError(f"client cap exceeded for seed {seed}: {client_sizes}")
        if force_info["missing_tx_tools"]:
            raise RuntimeError(f"missing labeled pool examples for Tx tools: {force_info['missing_tx_tools'][:10]}")

        per_seed[str(seed)] = {
            "owners": owners,
            "owner_counts": {client_id: sum(1 for owner in owners.values() if owner == client_id) for client_id in sorted(clients)},
            "client_sizes": client_sizes,
            "client_item_query_ids": {client_id: [item.query_id for item in items] for client_id, items in sorted(clients.items())},
            "client_item_sha256": {
                client_id: stable_hash([item_record(item) for item in items]) for client_id, items in sorted(clients.items())
            },
            "force_info": force_info,
        }
        save_path = args.output_dir / f"seed_{seed}" / "client_draws.json"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(per_seed[str(seed)], indent=2, sort_keys=True), encoding="utf-8")
        progress.log(
            "seed_done",
            seed=seed,
            client_sizes=client_sizes,
            owner_counts=per_seed[str(seed)]["owner_counts"],
            non_owner_mention_count_total=force_info["non_owner_mention_count_total"],
        )

    summary = {
        "paper_eligible": True,
        "repo_commit": commit,
        "config": {
            "groups": groups,
            "tx_group": TX_GROUP,
            "heldout_groups": HELDOUT_GROUPS,
            "seeds": seeds,
            "client_count": args.client_count,
            "max_items_per_client": args.max_items_per_client,
            "partition_mode": args.partition_mode,
            "target_items_per_tool": args.target_items_per_tool,
            "heldout_tools_sha256": heldout_sha,
            "candidate_stage": "e8_step1_ownership_and_forced_client_draws",
        },
        "heldout_sanity": heldout_sanity,
        "heldout_filter": heldout_filter,
        "tx_tools": tx_tools,
        "requested_tx_tools": requested_tx_tools,
        "excluded_tx_tools_no_single_tx_training_item": excluded_tx_tools,
        "tx_tool_count": len(tx_tools),
        "requested_tx_tool_count": len(requested_tx_tools),
        "excluded_tx_tool_count": len(excluded_tx_tools),
        "tx_query_ids": tx_query_ids,
        "tx_query_count": len(tx_query_ids),
        "per_seed": per_seed,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    progress.log("complete", output_dir=str(args.output_dir))
    print(json.dumps({"tx_tool_count": len(tx_tools), "tx_query_count": len(tx_query_ids), "seeds": seeds}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
