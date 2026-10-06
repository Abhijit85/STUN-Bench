#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_ROOT = Path(os.environ.get("FEDRAG_CANONICAL_ROOT", "<REPO_ROOT>")).resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from math_qa import JinaAIClient
from scripts.run_reranker_prompt_sweep import RoutedCandidate, run_prompt
from scripts.run_stabletoolbench_federated import (
    DEFAULT_MODEL_PATH,
    DEFAULT_STB_ROOT,
    ProgressLogger,
    assert_clean_tree,
    batched_query_embeddings,
    build_ranked_candidates,
    hash_package,
    load_local_backend,
    load_package_file,
    maybe_cuda_synchronize,
    package_to_candidates,
    resolve_local_embedder,
    save_json,
    summarize_rows,
    temporary_env,
)
from scripts.run_stabletoolbench_renderswap import decode_flat_payload, sha256_file, source_cell_path
from scripts.run_stabletoolbench_typing_isolation import load_eval_queries, load_seed_packages
from scripts.run_reranker_prompt_sweep import render_precautions, short_text
from synapse.knowledge.compendium import KnowledgePackage

DEFAULT_OUTPUT_DIR = CANONICAL_ROOT / "artifacts" / "results" / "stabletoolbench_frozen_renderswap_r1"
DEFAULT_SOURCE_SUMMARIES = {
    42: CANONICAL_ROOT / "artifacts" / "results" / "stabletoolbench_2x2_seed42_r3b" / "summary.json",
    123: Path("<REPO_ROOT>_runs/artifacts/results/stabletoolbench_2x2_seed123_r5/summary.json"),
    456: CANONICAL_ROOT / "artifacts" / "results" / "stabletoolbench_2x2_seed456_r2" / "summary.json",
}
SOURCE_PACKAGE_NAMES = {
    "typed_conflictlog": "typed_conflictlog",
    "flat_majority": "flat_majority",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen-candidate render swap for StableToolBench 2x2 cells.")
    parser.add_argument("--seeds", type=str, default="42,123,456")
    parser.add_argument("--rates", type=str, default="0,60")
    parser.add_argument("--merges", type=str, default="typed_conflictlog,flat_majority")
    parser.add_argument("--renders", type=str, default="typed,flat")
    parser.add_argument("--source-summary", action="append", default=[])
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


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_csv_int(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_source_summaries(args: argparse.Namespace, seeds: list[int]) -> dict[int, Path]:
    if not args.source_summary:
        return {seed: DEFAULT_SOURCE_SUMMARIES[seed] for seed in seeds}
    paths = [Path(item) for item in args.source_summary]
    if len(paths) == 1:
        return {seed: paths[0] for seed in seeds}
    if len(paths) != len(seeds):
        raise ValueError("--source-summary must be omitted, passed once, or passed once per seed")
    return {seed: path for seed, path in zip(seeds, paths)}


def source_package_path(packages_root: Path, seed: int, merge: str, rate: int) -> Path:
    seed_dir = packages_root / f"seed_{seed}" if (packages_root / f"seed_{seed}").exists() else packages_root
    return seed_dir / "renderswap_cache" / f"{SOURCE_PACKAGE_NAMES[merge]}_conflict_{rate:02d}.json"


def package_for_render(package: KnowledgePackage, render: str) -> tuple[KnowledgePackage, int]:
    if render == "typed":
        if any(str((artifact.structured_payload or {}).get("payload_mode")) == "flat" for artifact in package.artifacts):
            return decode_flat_payload(package)
        return package, 0
    if render == "flat":
        # Keep IDs and tool labels fixed, but collapse what the reranker sees into one JSON blob.
        rendered = []
        from synapse.knowledge.compendium import KnowledgeArtifact
        for artifact in package.artifacts:
            payload = artifact.structured_payload or {}
            serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True)
            metadata = dict(artifact.metadata or {})
            flat_payload = {
                "type": payload.get("type") or "usage_scenario",
                "payload_mode": "flat",
                "serialized_payload": serialized,
                "tool_description": serialized,
                "scenario_context": serialized,
                "precautions": [],
                "annex_summary": "flat_render_from_frozen_candidates",
            }
            rendered.append(KnowledgeArtifact(signature=artifact.signature, text=serialized, structured_payload=flat_payload, metadata=metadata, textgrad_variable=artifact.textgrad_variable))
        return KnowledgePackage(source_id=f"{package.source_id}_flat_render", artifacts=rendered, metadata=dict(package.metadata or {})), 0
    raise ValueError(f"unknown render: {render}")


def candidates_by_id(package: KnowledgePackage, jina_client: JinaAIClient, embed_model: str) -> dict[str, RoutedCandidate]:
    return {candidate.candidate_id: candidate for candidate in package_to_candidates(package, jina_client, embed_model)}


def build_frozen_candidates(base_package: KnowledgePackage, test_items, jina_client, embed_model: str, args, out_path: Path, source_recall: float) -> dict[str, Any]:
    if out_path.exists():
        payload = load_json(out_path)
        if f"{float(payload['recall_at_5']):.3f}" != f"{source_recall:.3f}":
            raise RuntimeError(f"cached frozen candidate recall mismatch: {out_path} cached={payload['recall_at_5']:.3f} source={source_recall:.3f}")
        return payload
    candidates = package_to_candidates(base_package, jina_client, embed_model)
    query_embeddings = batched_query_embeddings(jina_client, [item.query for item in test_items], embed_model) if test_items else []
    candidate_matrix = np.asarray([candidate.embedding for candidate in candidates], dtype=np.float32) if candidates else np.zeros((0, 0), dtype=np.float32)
    if candidate_matrix.size:
        norms = np.linalg.norm(candidate_matrix, axis=1)
        norms[norms == 0.0] = 1.0
        candidate_matrix = candidate_matrix / norms[:, None]
    rows = []
    for item, embedding in zip(test_items, query_embeddings):
        query_vector = np.asarray(embedding, dtype=np.float32)
        norm = float(np.linalg.norm(query_vector))
        if norm > 0.0:
            query_vector = query_vector / norm
        similarities = candidate_matrix @ query_vector if candidate_matrix.size else np.asarray([], dtype=np.float32)
        ranked, retrieval_pool_tools = build_ranked_candidates(candidates, similarities, args.retrieval_pool_size, args.top_k, args.retrieval_mode)
        score_by_id = {candidates[idx].candidate_id: float(similarities[idx]) for idx in np.argsort(-similarities).tolist()[: max(args.retrieval_pool_size, args.top_k)]} if candidate_matrix.size else {}
        rows.append({
            "query_id": item.query_id,
            "query_text": item.query,
            "group": item.group,
            "gold_parent_tools": item.gold_tools,
            "candidate_ids": [candidate.candidate_id for candidate in ranked],
            "candidate_tools": [candidate.parent_tool for candidate in ranked],
            "retrieval_pool_tools": retrieval_pool_tools,
            "scores": [score_by_id.get(candidate.candidate_id, 0.0) for candidate in ranked],
            "gold_in_top_k": any(tool in item.gold_tools for tool in retrieval_pool_tools),
        })
    recall = sum(1 for row in rows if row["gold_in_top_k"]) / len(rows) if rows else 0.0
    if f"{recall:.3f}" != f"{source_recall:.3f}":
        raise RuntimeError(f"frozen candidate recall mismatch: {out_path} frozen={recall:.3f} source={source_recall:.3f}")
    payload = {
        "paper_eligible": True,
        "retrieval_backend": "frozen_jina",
        "retrieval_mode": args.retrieval_mode,
        "retrieval_pool_size": args.retrieval_pool_size,
        "top_k": args.top_k,
        "recall_at_5": recall,
        "rows": rows,
    }
    save_json(out_path, payload)
    return payload


def render_flat_candidate(candidate: RoutedCandidate) -> RoutedCandidate:
    payload = {
        "tool": candidate.parent_tool,
        "when_to_use": candidate.when_to_use,
        "do_not_use_when": candidate.do_not_use_when,
        "source": candidate.provenance,
        "text": candidate.text,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return RoutedCandidate(
        candidate_id=candidate.candidate_id,
        label=candidate.label,
        parent_tool=candidate.parent_tool,
        when_to_use=[blob],
        do_not_use_when=[],
        text=blob,
        embedding=candidate.embedding,
        provenance=candidate.provenance,
    )


def rerank_frozen(name: str, rendered_candidates: dict[str, RoutedCandidate], frozen_payload: dict[str, Any], backend: Any, args, *, progress: ProgressLogger, seed: int, merge: str, rate: int, render: str) -> dict[str, Any]:
    rows = []
    total_latency = total_rerank = 0.0
    parse_failures = 0
    for idx, item in enumerate(frozen_payload["rows"], start=1):
        started = time.perf_counter()
        ranked = [rendered_candidates[cid] for cid in item["candidate_ids"] if cid in rendered_candidates]
        if render == "flat":
            ranked = [render_flat_candidate(candidate) for candidate in ranked]
        top_candidate = ranked[0] if ranked else None
        if top_candidate is None:
            rows.append({"query_id": item["query_id"], "query_text": item["query_text"], "group": item["group"], "gold_parent_tools": item["gold_parent_tools"], "candidate_ids": item["candidate_ids"], "predicted_tool": "", "routed_correctly": False, "gold_in_top_k": item["gold_in_top_k"], "parse_ok": False, "fallback_used": True, "rerank_s": 0.0, "total_s": time.perf_counter() - started})
            continue
        maybe_cuda_synchronize(backend)
        rerank_started = time.perf_counter()
        result = run_prompt(backend, args.reranker_variant, "toolbench", item["query_text"], ranked, [], top_candidate)
        maybe_cuda_synchronize(backend)
        rerank_s = time.perf_counter() - rerank_started
        total_s = time.perf_counter() - started
        total_latency += total_s
        total_rerank += rerank_s
        parse_failures += int(not result.parse_ok)
        rows.append({
            "query_id": item["query_id"],
            "query_text": item["query_text"],
            "group": item["group"],
            "gold_parent_tools": item["gold_parent_tools"],
            "candidate_ids": item["candidate_ids"],
            "top_candidate_ids": item["candidate_ids"],
            "top_candidates": item["candidate_tools"],
            "retrieval_pool_tools": item["retrieval_pool_tools"],
            "predicted_tool": result.predicted_tool,
            "predicted_candidate": result.predicted_candidate,
            "routed_correctly": result.predicted_tool in item["gold_parent_tools"],
            "gold_in_top_k": item["gold_in_top_k"],
            "parse_ok": result.parse_ok,
            "fallback_used": result.fallback_used,
            "rerank_s": rerank_s,
            "retrieval_s": 0.0,
            "total_s": total_s,
            "prompt_hash": result.prompt_hash,
            "render": render,
            "merge": merge,
            "rate": rate,
            "seed": seed,
        })
        if idx == 1 or idx % 25 == 0 or idx == len(frozen_payload["rows"]):
            progress.log("rerank_progress", seed=seed, merge=merge, rate=rate, render=render, completed_queries=idx, total_queries=len(frozen_payload["rows"]), running_accuracy=sum(1 for row in rows if row["routed_correctly"]) / len(rows))
    summary = summarize_rows(rows)
    summary.update({
        "arm": name,
        "seed": seed,
        "merge": merge,
        "rate": rate,
        "render": render,
        "recall_at_5": frozen_payload["recall_at_5"],
        "parse_failure_rate": parse_failures / len(rows) if rows else 0.0,
        "mean_latency_seconds": total_latency / len(rows) if rows else 0.0,
        "mean_rerank_seconds": total_rerank / len(rows) if rows else 0.0,
        "rows": rows,
    })
    return summary


def sign_test(flips_pos: int, flips_neg: int) -> float:
    n = flips_pos + flips_neg
    if n == 0:
        return 1.0
    k = min(flips_pos, flips_neg)
    # two-sided exact binomial with p=.5
    return min(1.0, 2.0 * sum(math_comb(n, i) for i in range(k + 1)) / (2 ** n))


def math_comb(n: int, k: int) -> int:
    import math
    return math.comb(n, k)


def aggregate(results: list[dict[str, Any]], renders: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for merge in sorted({r["merge"] for r in results}):
        out[merge] = {}
        for rate in sorted({r["rate"] for r in results if r["merge"] == merge}):
            out[merge][str(rate)] = {}
            for render in renders:
                subset = [r for r in results if r["merge"] == merge and r["rate"] == rate and r["render"] == render]
                accs = [float(r["accuracy"]) for r in subset]
                recs = [float(r["recall_at_5"]) for r in subset]
                out[merge][str(rate)][render] = {
                    "mean_accuracy": statistics.mean(accs) if accs else 0.0,
                    "sd_accuracy": statistics.stdev(accs) if len(accs) > 1 else 0.0,
                    "mean_recall_at_5": statistics.mean(recs) if recs else 0.0,
                    "sd_recall_at_5": statistics.stdev(recs) if len(recs) > 1 else 0.0,
                    "per_seed": {str(r["seed"]): {"accuracy": r["accuracy"], "recall_at_5": r["recall_at_5"]} for r in subset},
                }
            typed = [r for r in results if r["merge"] == merge and r["rate"] == rate and r["render"] == "typed"]
            flat = [r for r in results if r["merge"] == merge and r["rate"] == rate and r["render"] == "flat"]
            paired = {}
            for t in typed:
                f = next((x for x in flat if x["seed"] == t["seed"]), None)
                if not f:
                    continue
                t_by_q = {row["query_id"]: row for row in t["rows"]}
                f_by_q = {row["query_id"]: row for row in f["rows"]}
                pos = neg = same = 0
                for qid, tr in t_by_q.items():
                    fr = f_by_q[qid]
                    if tr["candidate_ids"] != fr["candidate_ids"]:
                        raise RuntimeError(f"candidate id mismatch in aggregate merge={merge} rate={rate} seed={t['seed']} query={qid}")
                    tc = bool(tr["routed_correctly"])
                    fc = bool(fr["routed_correctly"])
                    if tc and not fc:
                        pos += 1
                    elif fc and not tc:
                        neg += 1
                    else:
                        same += 1
                paired[str(t["seed"])] = {"typed_correct_flat_wrong": pos, "flat_correct_typed_wrong": neg, "same": same, "sign_test_p": sign_test(pos, neg)}
            out[merge][str(rate)]["paired"] = paired
    return out


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    commit, dirty = assert_clean_tree(allow_dirty=args.allow_dirty)
    progress = ProgressLogger(args.output_dir)
    progress.log("launch", git_commit=commit, dirty_entry_count=len(dirty))
    seeds = parse_csv_int(args.seeds)
    rates = parse_csv_int(args.rates)
    merges = parse_csv(args.merges)
    renders = parse_csv(args.renders)
    source_summaries = resolve_source_summaries(args, seeds)
    embedder = resolve_local_embedder()
    jina_client = JinaAIClient(api_keys=os.environ.get("JINA_API_KEY") and [os.environ["JINA_API_KEY"]] or [])
    results = []
    source_inputs = []
    with temporary_env({
        "JINA_LOCAL_EMBED_MODEL": embedder["model_path"],
        "JINA_LOCAL_EMBED_DEVICE": embedder["device"],
        "JINA_LOCAL_EMBED_LOCAL_ONLY": embedder["local_only"],
        "JINA_API_KEY": None,
    }):
        backend = load_local_backend(args.model_path)
        for seed in seeds:
            summary_path = source_summaries[seed]
            source_summary = load_json(summary_path)
            groups = source_summary["config"]["groups"]
            packages_root = Path(source_summary["config"]["load_packages_root"])
            if not packages_root.is_absolute():
                packages_root = (CANONICAL_ROOT / packages_root).resolve()
            _client_packages, tool_doc_package, tool_doc_hash = load_seed_packages(packages_root, seed)
            test_items = load_eval_queries(args.stb_root, groups, tool_doc_package)
            progress.log("seed_begin", seed=seed, query_count=len(test_items), source_summary=str(summary_path))
            for rate in rates:
                for merge in merges:
                    cell_path = source_cell_path(summary_path, merge, rate, seed)
                    cell = load_json(cell_path)
                    package_path = source_package_path(packages_root, seed, merge, rate)
                    package, package_hash = load_package_file(package_path)
                    expected_hash = str((cell.get("compendium") or {}).get("global_sha256") or cell.get("compendium_sha256") or "")
                    if expected_hash and package_hash != expected_hash:
                        raise RuntimeError(f"source compendium hash mismatch seed={seed} merge={merge} rate={rate}: cell={expected_hash} package={package_hash}")
                    progress.log("cell_loaded", seed=seed, rate=rate, merge=merge, cell=str(cell_path), package=str(package_path), package_sha256=package_hash)
                    frozen_path = args.output_dir / "candidates" / f"{merge}_conflict_{rate:02d}_seed_{seed}.json"
                    frozen = build_frozen_candidates(package, test_items, jina_client, args.embed_model, args, frozen_path, float(cell["recall_at_5"]))
                    source_inputs.append({"seed": seed, "rate": rate, "merge": merge, "source_cell_path": str(cell_path), "source_cell_sha256": sha256_file(cell_path), "source_package_path": str(package_path), "source_package_sha256": package_hash, "frozen_candidates_path": str(frozen_path), "frozen_candidates_sha256": sha256_file(frozen_path)})
                    rendered_maps = {}
                    decode_counts = {}
                    for render in renders:
                        render_pkg, decode_count = package_for_render(package, render)
                        rendered_maps[render] = candidates_by_id(render_pkg, jina_client, args.embed_model)
                        decode_counts[render] = decode_count
                    render_rows_by_name = {}
                    for render in renders:
                        progress.log("rerank_begin", seed=seed, rate=rate, merge=merge, render=render)
                        result = rerank_frozen(f"{merge}_{render}", rendered_maps[render], frozen, backend, args, progress=progress, seed=seed, merge=merge, rate=rate, render=render)
                        result.update({
                            "paper_eligible": True,
                            "git_commit": commit,
                            "source_cell_path": str(cell_path),
                            "source_cell_sha256": sha256_file(cell_path),
                            "source_compendium_sha256": package_hash,
                            "frozen_candidates_path": str(frozen_path),
                            "frozen_candidates_sha256": sha256_file(frozen_path),
                            "decode_roundtrip_mismatch": decode_counts[render],
                            "tool_doc_sha256": tool_doc_hash,
                            "embedder": embedder,
                            "retrieval_backend": "frozen_jina",
                            "retrieval_mode": args.retrieval_mode,
                            "reranker_variant": args.reranker_variant,
                            "metric_definition": {"correct": "predicted_tool in gold_parent_tools", "recall_at_5": "source frozen top-20 candidate list"},
                        })
                        out_path = args.output_dir / merge / f"conflict_{rate:02d}" / render / f"seed_{seed}.json"
                        save_json(out_path, result)
                        render_rows_by_name[render] = result["rows"]
                        results.append(result)
                        progress.log("rerank_done", seed=seed, rate=rate, merge=merge, render=render, accuracy=result["accuracy"], recall_at_5=result["recall_at_5"])
                    if len(renders) >= 2:
                        base = render_rows_by_name[renders[0]]
                        for other_render in renders[1:]:
                            other = render_rows_by_name[other_render]
                            for left, right in zip(base, other):
                                if left["query_id"] != right["query_id"] or left["candidate_ids"] != right["candidate_ids"]:
                                    raise RuntimeError(f"candidate_ids differ across renders seed={seed} merge={merge} rate={rate} query={left.get('query_id')}")
                        progress.log("candidate_identity_assert_pass", seed=seed, rate=rate, merge=merge, renders=renders)
    summary = {
        "paper_eligible": True,
        "config": {
            "git_commit": commit,
            "seeds": seeds,
            "rates": rates,
            "merges": merges,
            "renders": renders,
            "top_k": args.top_k,
            "retrieval_pool_size": args.retrieval_pool_size,
            "retrieval_mode": args.retrieval_mode,
            "reranker_variant": args.reranker_variant,
            "embed_model": args.embed_model,
            "model_path": args.model_path,
            "local_embedder": embedder,
        },
        "source_inputs": source_inputs,
        "aggregate": aggregate(results, renders),
    }
    save_json(args.output_dir / "summary.json", summary)
    progress.log("complete", output_dir=str(args.output_dir))
    print(json.dumps(summary["aggregate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
