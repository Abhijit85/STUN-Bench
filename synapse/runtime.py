from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from synapse.clients import ClientMetadata, SynapseClient, UnifiedQAClient
from synapse.config import ApiCredentials, FederationTopology, SynapseConfig
from synapse.edge import EdgeAggregator, EdgeConfig
from synapse.knowledge import KnowledgeArtifact, SynapseCompendium
from synapse.privacy.policies import PrivacyPolicy
from synapse.retrieval import RetrievalPlanner, RetrievalConfig
from synapse.server import SynapseServer, ServerConfig
from synapse.textgrad_support import TextGradSettings, textgrad_enabled_from_env


class SynapseRuntime:
    """
    High-level orchestrator for the SYNAPSE hierarchy.
    """

    def __init__(
        self,
        config: SynapseConfig,
        clients: Dict[str, SynapseClient],
        edges: Dict[str, EdgeAggregator],
        server: SynapseServer,
        retrieval_planner: Optional[RetrievalPlanner] = None,
        textgrad_settings: Optional[TextGradSettings] = None,
    ) -> None:
        self.config = config
        self.clients = clients
        self.edges = edges
        self.server = server
        self.retrieval_planner = retrieval_planner or RetrievalPlanner()
        self._last_snapshot: Optional[SynapseCompendium] = None
        self.textgrad_settings = textgrad_settings or TextGradSettings()

    @staticmethod
    def _dp_enabled_from_env(default: bool = True) -> bool:
        toggle = os.environ.get("SYNAPSE_ENABLE_DP")
        if toggle is None:
            return default
        return toggle.strip().lower() not in {"0", "false", "no", "off"}

    @classmethod
    def _resolve_dp_epsilon(cls, default: Optional[float]) -> Optional[float]:
        epsilon = default
        override = os.environ.get("SYNAPSE_DP_EPSILON")
        if override:
            try:
                epsilon = float(override)
            except ValueError:
                pass
        if not cls._dp_enabled_from_env(True):
            return None
        return epsilon

    @classmethod
    def build_local_runtime(
        cls,
        base_path: Path,
        credentials: ApiCredentials,
        client_count: Optional[int] = None,
    ) -> "SynapseRuntime":
        """
        Construct a runtime instance using repository data for MathQA and ScienceQA.
        """
        if client_count is None:
            env_client_count = os.environ.get("SYNAPSE_CLIENT_COUNT")
            if env_client_count:
                try:
                    client_count = max(1, int(env_client_count))
                except ValueError:
                    client_count = None
        if client_count is None:
            client_count = 2

        client_ids = [f"general-client-{i+1}" for i in range(client_count)]
        edge_clusters = {"edge-general": client_ids}

        topology = FederationTopology(
            client_ids=client_ids,
            edge_clusters=edge_clusters,
            central_server_id="synapse-central",
        )
        config = SynapseConfig(topology=topology, credentials=credentials)

        config.textgrad.enabled = config.textgrad.enabled or textgrad_enabled_from_env()
        eval_override = os.environ.get("SYNAPSE_TEXTGRAD_EVAL_ENGINE") or os.environ.get("TEXTGRAD_EVAL_ENGINE")
        test_override = os.environ.get("SYNAPSE_TEXTGRAD_TEST_ENGINE")
        aggregate_override = os.environ.get("SYNAPSE_TEXTGRAD_AGGREGATE")
        proximal_override = os.environ.get("SYNAPSE_TEXTGRAD_PROXIMAL")
        batch_override = os.environ.get("SYNAPSE_TEXTGRAD_BATCH_SIZE")
        max_steps_override = os.environ.get("SYNAPSE_TEXTGRAD_MAX_STEPS")

        if eval_override:
            config.textgrad.evaluation_engine_name = eval_override
        if test_override:
            config.textgrad.test_engine_name = test_override
        if aggregate_override:
            config.textgrad.aggregate_method = aggregate_override
        if proximal_override:
            config.textgrad.proximal_update = proximal_override.strip().lower() not in {"0", "false", "no", "off"}
        if batch_override:
            try:
                config.textgrad.batch_size = max(1, int(batch_override))
            except ValueError:
                pass
        if max_steps_override:
            try:
                parsed = int(max_steps_override)
            except ValueError:
                parsed = None
            config.textgrad.max_steps = parsed

        dp_epsilon: Optional[float]
        if config.enable_privacy:
            dp_epsilon = cls._resolve_dp_epsilon(1.0)
        else:
            dp_epsilon = None

        clients: Dict[str, SynapseClient] = {}

        for client_id in topology.client_ids:
            clients[client_id] = UnifiedQAClient(
                metadata=ClientMetadata(
                    client_id=client_id,
                    domain_tags=["math", "science", "multimodal"],
                    capabilities={"modality": "image+text"},
                ),
                math_compendium_path=base_path / "mathqa_tools_compendium.json",
                math_training_path=base_path / "train_new.json",
                science_compendium_path=base_path / "scienceqa_tools_compendium.json",
                science_dataset_path=base_path / "scienceqa_dataset.json",
                privacy_policy=PrivacyPolicy(dp_epsilon=dp_epsilon),
                textgrad_settings=config.textgrad,
            )

        edges: Dict[str, EdgeAggregator] = {
            "edge-general": EdgeAggregator(
                EdgeConfig(edge_id="edge-general", domains=["math", "science"]),
                textgrad_settings=config.textgrad,
            ),
        }

        server = SynapseServer(ServerConfig(server_id="synapse-central"))
        retrieval = RetrievalPlanner(RetrievalConfig(max_artifacts=6))

        return cls(
            config=config,
            clients=clients,
            edges=edges,
            server=server,
            retrieval_planner=retrieval,
            textgrad_settings=config.textgrad,
        )

    def run_round(self) -> None:
        """
        Execute a full round of client -> edge -> server knowledge propagation.
        """
        self._prepare_textgrad_round()

        for edge_id, client_ids in self.config.topology.edge_clusters.items():
            edge = self.edges[edge_id]
            packages = []
            for client_id in client_ids:
                client = self.clients[client_id]
                package = client.prepare_for_edge()
                if package.artifacts:
                    packages.append(package)
            if not packages:
                continue
            merged = edge.merge_packages(packages)
            if merged:
                self.server.ingest_from_edge(merged)

        self._last_snapshot = self.server.compendium

    def get_context_for_query(self, query: str, max_items: int = 5) -> List[KnowledgeArtifact]:
        """
        Retrieve the most relevant knowledge artifacts for a query using
        the latest global snapshot.
        """
        compendium = self.server.compendium
        artifacts = compendium.build_snapshot().artifacts
        planner = self.retrieval_planner or RetrievalPlanner(RetrievalConfig(max_artifacts=max_items))
        planner.config.max_artifacts = max_items
        return planner.select(query, artifacts)

    def export_snapshot(self, path: Path) -> None:
        """
        Persist the current compendium snapshot to disk.
        """
        snapshot = self.server.distribute_snapshot()
        payload = {
            "metadata": snapshot.metadata,
            "artifacts": [
                {
                    "signature": artifact.signature,
                    "text": artifact.text,
                    "structured_payload": artifact.structured_payload,
                    "metadata": artifact.metadata,
                }
                for artifact in snapshot.artifacts
            ],
        }
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    def summarize_round(self) -> Dict[str, object]:
        """
        Provide a lightweight summary of the current federation state.
        """
        snapshot = self.server.distribute_snapshot()
        return {
            "artifact_count": len(snapshot.artifacts),
            "version_history": self.server.version_history,
        }

    def plan_context_for_tool(self, query: str, tool_name: str) -> List[str]:
        """
        Generate textual snippets to augment a downstream tool prompt.
        """
        relevant_artifacts = self.get_context_for_query(query, max_items=5)
        snippets: List[str] = []
        for artifact in relevant_artifacts:
            if artifact.metadata.get("tool") and artifact.metadata["tool"] != tool_name:
                continue
            snippets.append(artifact.text)
        return snippets

    def _prepare_textgrad_round(self) -> None:
        """
        Ensure shared TextGrad settings are active before the round begins.
        """
        if not self.textgrad_settings.enabled:
            return

        self.textgrad_settings.ensure_engines()

        for edge in self.edges.values():
            edge.textgrad_settings = self.textgrad_settings

        for client in self.clients.values():
            configure = getattr(client, "set_textgrad_settings", None)
            if callable(configure):
                configure(self.textgrad_settings)
