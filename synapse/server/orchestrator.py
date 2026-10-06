from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from synapse.edge.aggregator import EdgeAggregator
from synapse.knowledge.compendium import KnowledgePackage, SynapseCompendium


@dataclass
class ServerConfig:
    """Configuration object for the central SYNAPSE server."""

    server_id: str = "synapse-central"
    enable_versioning: bool = True


class SynapseServer:
    """
    Central coordinator that receives consolidated knowledge packages
    from edge aggregators, updates the global compendium, and distributes
    new versions.
    """

    def __init__(self, config: Optional[ServerConfig] = None) -> None:
        self.config = config or ServerConfig()
        self.compendium = SynapseCompendium()
        self._versions: List[Dict[str, str]] = []

    def ingest_from_edge(self, package: KnowledgePackage) -> None:
        """
        Update the global compendium with artifacts from an edge package.
        """
        self.compendium.ingest(package)

        if self.config.enable_versioning:
            version_record = {
                "timestamp": datetime.utcnow().isoformat(),
                "edge_id": package.source_id,
                "artifact_count": str(len(package.artifacts)),
            }
            self._versions.append(version_record)

    def distribute_snapshot(self) -> KnowledgePackage:
        """
        Produce a snapshot package that can be sent to edges or clients.
        """
        snapshot = self.compendium.build_snapshot()
        snapshot.source_id = self.config.server_id
        return snapshot

    @property
    def version_history(self) -> List[Dict[str, str]]:
        """Lightweight access to recorded ingestion events."""
        return list(self._versions)
