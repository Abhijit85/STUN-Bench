#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class BenchmarkProvenance:
    benchmark: str
    source_path: str
    source_kind: str
    build_date: str
    commit: str = "unknown"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExperienceRecord:
    benchmark: str
    query_id: str
    query_text: str
    gold_tools: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExperienceBundle:
    provenance: BenchmarkProvenance
    records: list[ExperienceRecord]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provenance": self.provenance.to_dict(),
            "metadata": self.metadata,
            "records": [record.to_dict() for record in self.records],
        }
