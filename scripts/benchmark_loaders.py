#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_DATASETS = REPO_ROOT / "external_datasets"


@dataclass
class DatasetAvailability:
    name: str
    root: Path
    present: bool
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "root": str(self.root),
            "present": self.present,
            "notes": list(self.notes),
        }


def tau_bench_root() -> Path:
    return EXTERNAL_DATASETS / "tau_bench" / "tau-bench-main"


def stabletoolbench_root() -> Path:
    return EXTERNAL_DATASETS / "StableToolBench"


def toolbench_instruction_root() -> Path:
    return EXTERNAL_DATASETS / "toolbench_hf" / "instruction"


def nq_open_root() -> Path:
    return EXTERNAL_DATASETS / "nq_open"


def livemcpbench_root() -> Path:
    return EXTERNAL_DATASETS / "LiveMCPBench"


def check_tau_bench() -> DatasetAvailability:
    root = tau_bench_root()
    required = [root / "run.py", root / "historical_trajectories" / "gpt-4o-retail.json"]
    missing = [str(path) for path in required if not path.exists()]
    return DatasetAvailability(
        name="tau_bench",
        root=root,
        present=not missing,
        notes=missing or ["tau-bench retail bridge assets present"],
    )


def check_stabletoolbench() -> DatasetAvailability:
    root = stabletoolbench_root()
    required = [
        root / "solvable_queries" / "test_instruction" / "G1_instruction.json",
        toolbench_instruction_root() / "G1_query.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    return DatasetAvailability(
        name="stabletoolbench",
        root=root,
        present=not missing,
        notes=missing or ["StableToolBench test split and ToolBench training instructions present"],
    )


def check_nq_open() -> DatasetAvailability:
    root = nq_open_root()
    required = [root / "train.jsonl", root / "dev.jsonl"]
    missing = [str(path) for path in required if not path.exists()]
    notes = missing or ["NQ-Open train/dev assets present"]
    if not missing:
        wiki_index = root / "indexes"
        if not wiki_index.exists():
            notes.append("Wikipedia/KILT retrieval indexes still need to be built or linked")
    return DatasetAvailability(name="nq_open", root=root, present=not missing, notes=notes)


def check_livemcpbench() -> DatasetAvailability:
    root = livemcpbench_root()
    normalized_required = [root / "manifests" / "tools.json", root / "trajectories" / "all_annotations.json"]
    raw_required = [root / "tools" / "LiveMCPTool" / "tools.json", root / "annotated_data" / "all_annotations.json"]
    if all(path.exists() for path in normalized_required):
        return DatasetAvailability(
            name="livemcpbench",
            root=root,
            present=True,
            notes=["LiveMCPBench normalized manifests and trajectories present"],
        )
    if all(path.exists() for path in raw_required):
        return DatasetAvailability(
            name="livemcpbench",
            root=root,
            present=True,
            notes=["LiveMCPBench raw repo assets present; run scripts/prepare_livemcpbench_assets.py for normalized manifests/trajectories"],
        )
    missing = [str(path) for path in raw_required if not path.exists()]
    return DatasetAvailability(name="livemcpbench", root=root, present=False, notes=missing)


def benchmark_availability() -> dict[str, dict[str, Any]]:
    checks = [check_tau_bench(), check_stabletoolbench(), check_nq_open(), check_livemcpbench()]
    return {check.name: check.to_dict() for check in checks}


def require_dataset(name: str) -> Path:
    mapping = {
        "tau_bench": check_tau_bench,
        "stabletoolbench": check_stabletoolbench,
        "nq_open": check_nq_open,
        "livemcpbench": check_livemcpbench,
    }
    if name not in mapping:
        raise KeyError(f"Unknown dataset: {name}")
    status = mapping[name]()
    if not status.present:
        raise FileNotFoundError(f"{name} assets unavailable: {status.notes}")
    return status.root
