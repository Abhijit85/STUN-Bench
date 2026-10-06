from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class KnowledgeArtifact:
    """
    Atomic unit of knowledge shared within SYNAPSE.

    Attributes:
        signature: Deterministic identifier for deduplication.
        text: Human-readable explanation or exemplar.
        structured_payload: Optional symbolic representation (triples, code, etc.).
    """

    signature: str
    text: str
    structured_payload: Optional[Dict[str, object]] = None
    metadata: Dict[str, object] = field(default_factory=dict)
    textgrad_variable: Optional[Any] = None


@dataclass
class KnowledgePackage:
    """
    Bundle of knowledge artifacts transmitted between tiers.
    """

    source_id: str
    artifacts: List[KnowledgeArtifact] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, object] = field(default_factory=dict)


class SynapseCompendium:
    """
    Global knowledge store maintaining curated artifacts.
    """

    def __init__(self) -> None:
        self._artifacts: Dict[str, KnowledgeArtifact] = {}
        self._sources: Dict[str, List[str]] = {}

    def ingest(self, package: KnowledgePackage) -> None:
        """
        Merge artifacts from a package, replacing prior entries with the same signature.
        """
        for artifact in package.artifacts:
            self._artifacts[artifact.signature] = artifact
            self._sources.setdefault(artifact.signature, []).append(package.source_id)

    def build_snapshot(self) -> KnowledgePackage:
        """
        Generate a snapshot KnowledgePackage reflecting the current global state.
        """
        return KnowledgePackage(
            source_id="synapse-compendium",
            artifacts=list(self._artifacts.values()),
            metadata={"artifact_count": len(self._artifacts)},
        )

    def sources_for(self, signature: str) -> List[str]:
        return list(self._sources.get(signature, []))

    def __len__(self) -> int:
        return len(self._artifacts)

    def query_by_domain(self, domain: str) -> List[KnowledgeArtifact]:
        return [
            artifact
            for artifact in self._artifacts.values()
            if artifact.metadata.get("domain") == domain or artifact.metadata.get("tool") == domain
        ]

    def to_dict(self) -> Dict[str, object]:
        """
        Serialize compendium contents into a JSON-friendly structure.
        """
        return {
            "artifacts": [
                {
                    "signature": artifact.signature,
                    "text": artifact.text,
                    "structured_payload": artifact.structured_payload,
                    "metadata": artifact.metadata,
                    "sources": self._sources.get(artifact.signature, []),
                }
                for artifact in self._artifacts.values()
            ]
        }
