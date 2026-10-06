from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from synapse.textgrad_support import TextGradSettings


@dataclass
class FederationTopology:
    """
    Describes the hierarchical layout of the SYNAPSE federation.
    """

    client_ids: List[str]
    edge_clusters: Dict[str, List[str]]  # edge_id -> client_ids
    central_server_id: str = "synapse-central"


@dataclass
class ApiCredentials:
    """
    Stores API keys and endpoints required by SYNAPSE components.
    """

    lambda_api_key: str
    jina_api_key: str
    mongo_uri: str
    lambda_api_base: str = "https://openrouter.ai/api/v1/chat/completions"
    jina_embed_url: str = "https://api.jina.ai/v1/embeddings"
    jina_rerank_url: str = "https://api.jina.ai/v1/rerank"


@dataclass
class SynapseConfig:
    """
    Top-level configuration container consumed by runtime harnesses.
    """

    topology: FederationTopology
    credentials: ApiCredentials
    enable_privacy: bool = True
    snapshot_interval: int = 1  # number of rounds between server snapshots
    textgrad: TextGradSettings = field(default_factory=TextGradSettings)
