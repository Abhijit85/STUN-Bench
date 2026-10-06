from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, List, Optional

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except Exception:  # pragma: no cover - optional local reranker dependency
    torch = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None

from synapse.knowledge.compendium import KnowledgeArtifact


@dataclass
class RetrievalConfig:
    """Configuration for dynamic context retrieval."""

    max_artifacts: int = 5
    enable_symbolic_queries: bool = True
    local_reranker_model: Optional[str] = field(
        default_factory=lambda: os.environ.get("SYNAPSE_LOCAL_RERANK_MODEL")
        or os.environ.get("SYNAPSE_RERANK_MODEL")
    )
    local_reranker_device: Optional[str] = field(
        default_factory=lambda: os.environ.get("SYNAPSE_LOCAL_RERANK_DEVICE")
        or os.environ.get("SYNAPSE_RERANK_DEVICE")
    )
    local_reranker_batch_size: int = field(
        default_factory=lambda: int(os.environ.get("SYNAPSE_LOCAL_RERANK_BATCH_SIZE", "8"))
    )
    local_reranker_max_length: int = field(
        default_factory=lambda: int(os.environ.get("SYNAPSE_LOCAL_RERANK_MAX_LENGTH", "512"))
    )
    lexical_weight: float = field(
        default_factory=lambda: float(os.environ.get("SYNAPSE_RETRIEVAL_LEXICAL_WEIGHT", "0.15"))
    )
    retrieval_profile: str = field(
        default_factory=lambda: os.environ.get("SYNAPSE_RETRIEVAL_PROFILE", "default")
    )


class RetrievalPlanner:
    """
    Coordinates retrieval of relevant knowledge artifacts for a new query.
    """

    def __init__(self, config: Optional[RetrievalConfig] = None) -> None:
        self.config = config or RetrievalConfig()
        self._reranker_tokenizer = None
        self._reranker_model = None
        self._reranker_device = None

    def select(self, query: str, artifacts: Iterable[KnowledgeArtifact]) -> List[KnowledgeArtifact]:
        """
        Rank artifacts using a local reranker when configured, otherwise
        fall back to lexical matching plus light symbolic bonuses.
        """
        ranked_artifacts = list(artifacts)
        if not ranked_artifacts:
            return []

        rerank_scores = self._rerank_scores(query, ranked_artifacts)
        if rerank_scores is None:
            ranked = sorted(
                ranked_artifacts,
                key=lambda art: self._score_artifact(query, art),
                reverse=True,
            )
            return ranked[: self.config.max_artifacts]

        ranked = sorted(
            zip(ranked_artifacts, rerank_scores),
            key=lambda item: item[1] + self.config.lexical_weight * self._score_artifact(query, item[0]),
            reverse=True,
        )
        return [artifact for artifact, _ in ranked[: self.config.max_artifacts]]

    def _tokenize(self, text: str) -> Counter:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        return Counter(tokens)

    def _score_artifact(self, query: str, artifact: KnowledgeArtifact) -> float:
        query_tokens = self._tokenize(query)
        artifact_tokens = self._tokenize(artifact.text)

        overlap = sum((query_tokens & artifact_tokens).values())
        length_norm = max(len(artifact_tokens), 1)
        token_score = overlap / length_norm

        profile = (self.config.retrieval_profile or "default").strip().lower()
        structured_bonus = 0.0
        artifact_type = None
        if self.config.enable_symbolic_queries and artifact.structured_payload:
            payload = artifact.structured_payload
            if isinstance(payload, dict):
                artifact_type = payload.get("type")
                for value in payload.values():
                    if isinstance(value, str) and value.lower() in query.lower():
                        structured_bonus += 0.5
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, str) and item.lower() in query.lower():
                                structured_bonus += 0.2

        if isinstance(artifact_type, str):
            if artifact_type == "usage_scenario":
                structured_bonus += 0.6 if profile == "paperlike" else 0.15
            elif artifact_type == "training_example" and profile == "paperlike":
                structured_bonus -= 0.35

        if profile == "paperlike":
            scenario = artifact.metadata.get("scenario")
            if isinstance(scenario, str) and scenario.strip():
                scenario_tokens = self._tokenize(scenario)
                scenario_overlap = sum((query_tokens & scenario_tokens).values())
                if scenario_overlap:
                    structured_bonus += 0.35 * scenario_overlap / max(len(scenario_tokens), 1)

        return token_score + structured_bonus

    def _ensure_reranker(self) -> bool:
        model_name = self.config.local_reranker_model
        if not model_name:
            return False
        if self._reranker_model is not None and self._reranker_tokenizer is not None:
            return True
        if torch is None or AutoTokenizer is None or AutoModelForSequenceClassification is None:
            return False

        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True)

        device_name = self.config.local_reranker_device
        if not device_name:
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        if device_name.startswith("cuda") and not torch.cuda.is_available():
            device_name = "cpu"

        device = torch.device(device_name)
        model.to(device)
        model.eval()

        self._reranker_tokenizer = tokenizer
        self._reranker_model = model
        self._reranker_device = device
        return True

    def _rerank_scores(self, query: str, artifacts: List[KnowledgeArtifact]) -> Optional[List[float]]:
        if not self._ensure_reranker():
            return None

        texts = [artifact.text for artifact in artifacts]
        scores: List[float] = []
        batch_size = max(1, self.config.local_reranker_batch_size)

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded = self._reranker_tokenizer(
                [query] * len(batch_texts),
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.config.local_reranker_max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self._reranker_device) for key, value in encoded.items()}

            with torch.no_grad():
                logits = self._reranker_model(**encoded).logits

            if logits.ndim == 2 and logits.shape[-1] == 1:
                batch_scores = logits[:, 0]
            elif logits.ndim == 2:
                batch_scores = logits[:, -1]
            else:
                batch_scores = logits.view(-1)

            scores.extend(float(value) for value in batch_scores.detach().cpu())

        return scores
