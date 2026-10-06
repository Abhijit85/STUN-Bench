#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from math_qa import JinaAIClient
from scripts.prepare_stabletoolbench_cross_client_e8 import TX_GROUP, force_single_owner_draws, item_record
from scripts.run_reranker_prompt_sweep import run_prompt
from scripts.run_stabletoolbench_federated import (
    DEFAULT_MODEL_PATH,
    DEFAULT_STB_ROOT,
    DEFAULT_TOOLBENCH_INSTRUCTION_DIR,
    GROUPS,
    ProgressLogger,
    apply_junk_filter,
    assert_clean_tree,
    batched_query_embeddings,
    build_client_package,
    build_doc_package,
    build_ranked_candidates,
    build_tool_registry,
    assign_clients,
    candidate_selection_diagnostics,
    combine_packages,
    filter_eval_queries,
    filter_experience_items,
    limit_client_items,
    load_local_backend,
    load_stabletoolbench_queries,
    load_toolbench_training_items,
    maybe_cuda_synchronize,
    package_to_candidates,
    resolve_local_embedder,
    save_json,
    stable_hash,
    summarize_rows,
    temporary_env,
)
from scripts.run_stabletoolbench_heldout import HELDOUT_GROUPS, filter_heldout_pool, heldout_tool_set


DEFAULT_INPUT_DIR = REPO_ROOT / "artifacts" / "results" / "stabletoolbench_cross_client_e8_inputs_r2"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "results" / "stabletoolbench_cross_client_e8_smoke_r1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E8 step 2 smoke: non-owner local-only and docs-only on 20 Tx queries.")
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    p.add_argument("--stb-root", type=Path, default=DEFAULT_STB_ROOT)
    p.add_argument("--toolbench-instruction-dir", type=Path, default=DEFAULT_TOOLBENCH_INSTRUCTION_DIR)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--query-limit", type=int, default=20)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--retrieval-pool-size", type=int, default=200)
    p.add_argument("--retrieval-mode", default="distinct_tool_topk")
    p.add_argument("--reranker-variant", default="V3")
    p.add_argument("--embed-model", default="jina-embeddings-v2-base-en")
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--allow-dirty", action="store_true")
    return p.parse_args()


def evaluate_package(name: str, package, test_items, jina_client, embed_model: str, backend: Any, args: argparse.Namespace, progress: ProgressLogger, *, seed: int, client_id: str | None = None) -> dict[str, Any]:
    progress.log("arm_prepare_begin", arm=name, seed=seed, client_id=client_id, query_count=len(test_items), artifact_count=len(package.artifacts))
    candidates = package_to_candidates(package, jina_client, embed_model)
    candidate_matrix = None
    if candidates:
        import numpy as np

        candidate_matrix = np.asarray([candidate.embedding for candidate in candidates], dtype=np.float32)
        norms = np.linalg.norm(candidate_matrix, axis=1)
        norms[norms == 0.0] = 1.0
        candidate_matrix = candidate_matrix / norms[:, None]
    query_embeddings = batched_query_embeddings(jina_client, [item.query for item in test_items], embed_model) if test_items else []
    rows = []
    for idx, (item, embedding) in enumerate(zip(test_items, query_embeddings), start=1):
        import numpy as np

        q = np.asarray(embedding, dtype=np.float32)
        qn = float(np.linalg.norm(q))
        if qn > 0.0:
            q = q / qn
        similarities = candidate_matrix @ q if candidate_matrix is not None and candidate_matrix.size else np.asarray([], dtype=np.float32)
        ranked, retrieval_pool_tools = build_ranked_candidates(candidates, similarities, args.retrieval_pool_size, args.top_k, args.retrieval_mode)
        diag = candidate_selection_diagnostics(candidates, similarities, ranked, args.top_k, args.retrieval_pool_size, args.retrieval_mode)
        if diag["candidate_shortfall"]:
            raise RuntimeError(f"{name} seed {seed} client {client_id} query {item.query_id} shortfall: {diag}")
        top_candidate = ranked[0] if ranked else None
        maybe_cuda_synchronize(backend)
        if top_candidate is None:
            pred = ""
            parse_ok = False
            fallback = True
            prompt_hash = ""
        else:
            result = run_prompt(backend, args.reranker_variant, "toolbench", item.query, ranked, [], top_candidate)
            pred = result.predicted_tool
            parse_ok = result.parse_ok
            fallback = result.fallback_used
            prompt_hash = result.prompt_hash
        maybe_cuda_synchronize(backend)
        rows.append({
            "query_id": item.query_id,
            "query_text": item.query,
            "group": item.group,
            "gold_parent_tools": item.gold_tools,
            "gold_tools": item.gold_tools,
            "client_id": client_id,
            "arm": name,
            "predicted_tool": pred,
            "routed_correctly": pred in item.gold_tools,
            "correct": pred in item.gold_tools,
            "gold_in_top_k": any(candidate.parent_tool in item.gold_tools for candidate in ranked),
            "top_candidate_ids": [candidate.parent_tool for candidate in ranked],
            "retrieval_pool_tools": retrieval_pool_tools,
            "parse_ok": parse_ok,
            "fallback_used": fallback,
            "prompt_hash": prompt_hash,
            **diag,
        })
        if idx == 1 or idx == len(test_items):
            progress.log("arm_progress", arm=name, seed=seed, client_id=client_id, completed_queries=idx, total_queries=len(test_items))
    out = summarize_rows(rows)
    out.update({"arm": name, "seed": seed, "client_id": client_id, "rows": rows})
    return out


def main() -> int:
    load_dotenv(REPO_ROOT / ".env")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    commit, dirty = assert_clean_tree(allow_dirty=args.allow_dirty)
    progress = ProgressLogger(args.output_dir)
    progress.log("launch", git_commit=commit, dirty_entry_count=len(dirty), input_dir=str(args.input_dir))

    input_summary = json.loads((args.input_dir / "summary.json").read_text(encoding="utf-8"))
    seed_rec = input_summary["per_seed"][str(args.seed)]
    owners: dict[str, str] = seed_rec["owners"]
    query_ids = set(input_summary["tx_query_ids"][: args.query_limit])

    queries = load_stabletoolbench_queries(args.stb_root, GROUPS)
    train_items = load_toolbench_training_items(args.toolbench_instruction_dir)
    registry = build_tool_registry(queries + train_items)
    registry, _ = apply_junk_filter(registry)
    queries, _ = filter_eval_queries(queries, registry)
    train_items, _ = filter_experience_items(train_items, registry)
    heldout_tools, _ = heldout_tool_set(queries, HELDOUT_GROUPS)
    train_items, _ = filter_heldout_pool(train_items, set(heldout_tools))
    test_items = [item for item in queries if item.query_id in query_ids and item.group == TX_GROUP]
    if len(test_items) != min(args.query_limit, len(input_summary["tx_query_ids"])):
        raise RuntimeError(f"expected {args.query_limit} Tx queries, found {len(test_items)}")

    embedder = resolve_local_embedder()
    jina_client = JinaAIClient(api_keys=[])
    with temporary_env({
        "JINA_LOCAL_EMBED_MODEL": embedder["model_path"],
        "JINA_LOCAL_EMBED_DEVICE": embedder["device"],
        "JINA_LOCAL_EMBED_LOCAL_ONLY": embedder["local_only"],
        "JINA_API_KEY": None,
    }):
        cfg = input_summary["config"]
        client_count = int(cfg.get("client_count", len(seed_rec["client_item_query_ids"])))
        max_items_per_client = int(cfg.get("max_items_per_client", 5000))
        partition_mode = cfg.get("partition_mode", "category")
        target_items_per_tool = int(cfg.get("target_items_per_tool", 8))
        base_clients = assign_clients(train_items, client_count, partition_mode, args.seed)
        capped_clients = limit_client_items(base_clients, max_items_per_client, args.seed)
        client_items, _force_info = force_single_owner_draws(
            capped_clients,
            train_items,
            set(owners),
            owners,
            seed=args.seed,
            max_items_per_client=max_items_per_client,
            target_items_per_tool=target_items_per_tool,
        )
        for client_id, items in client_items.items():
            digest = stable_hash([item_record(item) for item in items])
            if digest != seed_rec["client_item_sha256"][client_id]:
                raise RuntimeError(f"client draw hash mismatch for {client_id}: {digest} != {seed_rec['client_item_sha256'][client_id]}")
        tool_doc_package, tool_doc_hash = build_doc_package(registry)
        backend = load_local_backend(args.model_path)
        docs_result = evaluate_package("docs_only", tool_doc_package, test_items, jina_client, args.embed_model, backend, args, progress, seed=args.seed)
        docs_result.update({"tool_doc_sha256": tool_doc_hash})
        save_json(args.output_dir / f"seed_{args.seed}" / "docs_only.json", docs_result)

        local_rows = []
        client_results = []
        for client_id, items in sorted(client_items.items()):
            package, package_hash = build_client_package(client_id, items, registry, jina_client, args.embed_model)
            combined, combined_hash = combine_packages(f"{client_id}_with_docs", [tool_doc_package, package])
            client_queries = []
            for item in test_items:
                # G1-instruction is single-tool in this stratum; skip owner, score non-owners.
                gold = item.gold_tools[0]
                if owners[gold] != client_id:
                    client_queries.append(item)
            result = evaluate_package("local_only_non_owner", combined, client_queries, jina_client, args.embed_model, backend, args, progress, seed=args.seed, client_id=client_id)
            result.update({"client_package_sha256": package_hash, "combined_sha256": combined_hash})
            save_json(args.output_dir / f"seed_{args.seed}" / f"local_only_{client_id}.json", result)
            client_results.append(result)
            local_rows.extend(result["rows"])

        local_summary = summarize_rows(local_rows)
        local_summary.update({"arm": "local_only_non_owner", "seed": args.seed, "rows": local_rows})
        save_json(args.output_dir / f"seed_{args.seed}" / "local_only_non_owner.json", local_summary)
        summary = {
            "paper_eligible": False,
            "smoke_test": True,
            "repo_commit": commit,
            "config": {
                "input_dir": str(args.input_dir),
                "input_repo_commit": input_summary["repo_commit"],
                "seed": args.seed,
                "query_limit": args.query_limit,
                "top_k": args.top_k,
                "retrieval_pool_size": args.retrieval_pool_size,
                "retrieval_mode": args.retrieval_mode,
                "docs_only_byte_identity_guard": "client draw hashes matched input summary; docs-only uses one tool-doc package for all non-owner comparisons",
            },
            "docs_only": {k: docs_result[k] for k in ("accuracy", "recall_at_5", "parse_failure_rate") if k in docs_result},
            "local_only_non_owner": {k: local_summary[k] for k in ("accuracy", "recall_at_5", "parse_failure_rate") if k in local_summary},
            "non_owner_rows": len(local_rows),
        }
        save_json(args.output_dir / "summary.json", summary)
    progress.log("complete", output_dir=str(args.output_dir))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
