#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_stabletoolbench_federated import (
    DEFAULT_MODEL_PATH,
    DEFAULT_STB_ROOT,
    JinaAIClient,
    KnowledgeArtifact,
    KnowledgePackage,
    evaluate_reranker_arm,
    filter_eval_queries,
    load_local_backend,
    load_stabletoolbench_queries,
    parse_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay StableToolBench evaluation from persisted packages.")
    parser.add_argument("--packages-dir", type=Path, required=True)
    parser.add_argument("--package-file", type=str, default="synapse_with_docs.json")
    parser.add_argument("--groups", type=str, default="G1_instruction,G1_tool,G1_category,G2_instruction,G2_category,G3_instruction")
    parser.add_argument("--stb-root", type=Path, default=DEFAULT_STB_ROOT)
    parser.add_argument("--embed-model", type=str, default="jina-embeddings-v2-base-en")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retrieval-pool-size", type=int, default=5)
    parser.add_argument("--retrieval-mode", choices=["entry_topk", "distinct_tool_topk", "union_doc_scenario"], default="distinct_tool_topk")
    parser.add_argument("--reranker-variant", choices=["V1", "V3"], default="V3")
    parser.add_argument("--output-file", type=Path, required=True)
    return parser.parse_args()


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
    return KnowledgePackage(source_id=package.get("source_id", "replayed"), artifacts=artifacts, metadata=package.get("metadata"))


def build_per_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["group"]].append(row)
    summary: dict[str, Any] = {}
    for group, bucket in sorted(grouped.items()):
        summary[group] = {
            "count": len(bucket),
            "accuracy": sum(1 for row in bucket if row["routed_correctly"]) / len(bucket),
            "recall_at_5": sum(1 for row in bucket if row["gold_in_top_k"]) / len(bucket),
        }
    return summary


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    args = parse_args()
    package_payload = json.loads((args.packages_dir / args.package_file).read_text(encoding="utf-8"))
    package = deserialize_package(package_payload)
    groups = parse_csv(args.groups)
    test_items = load_stabletoolbench_queries(args.stb_root, groups)
    available_tools = {
        str((artifact.metadata or {}).get("tool") or "").strip()
        for artifact in package.artifacts
        if str((artifact.metadata or {}).get("tool") or "").strip()
    }
    registry = {tool: object() for tool in available_tools}
    test_items, filter_info = filter_eval_queries(test_items, registry)

    backend = load_local_backend(args.model_path)
    jina_keys = os.environ.get("JINA_API_KEY") and [os.environ["JINA_API_KEY"]] or []
    jina_client = JinaAIClient(api_keys=jina_keys)
    result = evaluate_reranker_arm(
        "replay",
        package,
        test_items,
        jina_client,
        args.embed_model,
        backend,
        args.top_k,
        args.retrieval_pool_size,
        args.retrieval_mode,
        args.reranker_variant,
    )
    result["packages_dir"] = str(args.packages_dir)
    result["package_file"] = args.package_file
    result["package_sha256"] = package_payload.get("package_sha256")
    result["eval_filter"] = filter_info
    result["per_group"] = build_per_group(result["rows"])
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "accuracy": result["accuracy"],
        "recall_at_5": result["recall_at_5"],
        "per_group": result["per_group"],
        "output_file": str(args.output_file),
    }, indent=2))


if __name__ == "__main__":
    main()
