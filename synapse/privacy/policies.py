from __future__ import annotations

import math
import random
import re
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional

from synapse.knowledge.compendium import KnowledgeArtifact


def _laplace(scale: float) -> float:
    """
    Sample noise from a Laplace(0, scale) distribution using inverse transform sampling.
    """
    u = random.random() - 0.5
    return -scale * math.copysign(1.0, u) * math.log(1 - 2 * abs(u))


@dataclass
class PrivacyPolicy:
    """
    Redaction and differential privacy controls applied before sharing artifacts.
    """

    redact_sensitive_metadata: bool = True
    drop_pii_text: bool = True
    dp_epsilon: Optional[float] = None
    dp_clip_abs: float = 1.0
    adaptive_text_noise: bool = True
    adaptive_digit_weight: float = 0.6
    adaptive_length_weight: float = 0.3
    adaptive_upper_weight: float = 0.2
    adaptive_title_weight: float = 0.1
    adaptive_probability_multiplier: float = 1.0
    adaptive_distortion_multiplier: float = 1.0

    def __post_init__(self) -> None:
        self._load_env_overrides()

    def _load_env_overrides(self) -> None:
        env = os.environ

        def _flag(name: str, default: bool) -> bool:
            value = env.get(name)
            if value is None:
                return default
            return value.strip().lower() in {"1", "true", "yes", "on"}

        if "SYNAPSE_ADAPTIVE_TEXT_NOISE" in env:
            self.adaptive_text_noise = _flag("SYNAPSE_ADAPTIVE_TEXT_NOISE", self.adaptive_text_noise)

        for attr, var in (
            ("dp_clip_abs", "SYNAPSE_DP_CLIP_ABS"),
            ("adaptive_digit_weight", "SYNAPSE_ADAPTIVE_DIGIT_WEIGHT"),
            ("adaptive_length_weight", "SYNAPSE_ADAPTIVE_LENGTH_WEIGHT"),
            ("adaptive_upper_weight", "SYNAPSE_ADAPTIVE_UPPER_WEIGHT"),
            ("adaptive_title_weight", "SYNAPSE_ADAPTIVE_TITLE_WEIGHT"),
            ("adaptive_probability_multiplier", "SYNAPSE_ADAPTIVE_PROBABILITY_MULT"),
            ("adaptive_distortion_multiplier", "SYNAPSE_ADAPTIVE_DISTORT_MULT"),
        ):
            value = env.get(var)
            if value is None:
                continue
            try:
                setattr(self, attr, float(value))
            except ValueError:
                continue

    def enforce(self, artifacts: Iterable[KnowledgeArtifact]) -> List[KnowledgeArtifact]:
        """
        Apply privacy protections (PII redaction and optional DP noise).
        """
        sanitized: List[KnowledgeArtifact] = []
        for artifact in artifacts:
            text = artifact.text
            if self.drop_pii_text and self._looks_like_pii(text):
                continue

            metadata = artifact.metadata or {}
            if self.redact_sensitive_metadata:
                filtered_meta = {
                    key: value
                    for key, value in metadata.items()
                    if not key.lower().startswith("pii")
                }
            else:
                filtered_meta = metadata

            sanitized.append(
                KnowledgeArtifact(
                    signature=artifact.signature,
                    text=text,
                    structured_payload=artifact.structured_payload,
                    metadata=filtered_meta,
                )
            )

        if self.dp_epsilon and self.dp_epsilon > 0:
            sanitized = self._apply_dp_noise(sanitized)
        return sanitized

    def _apply_dp_noise(self, artifacts: Iterable[KnowledgeArtifact]) -> List[KnowledgeArtifact]:
        delta_m = max(float(self.dp_clip_abs), 0.0)
        scale = delta_m / max(self.dp_epsilon, 1e-6)
        privatized: List[KnowledgeArtifact] = []
        for artifact in artifacts:
            metadata = dict(artifact.metadata or {})
            for key, value in metadata.items():
                if isinstance(value, (int, float)):
                    metadata[key] = self._clip_numeric(value) + _laplace(scale)

            payload = artifact.structured_payload
            if isinstance(payload, dict):
                privatized_payload = {}
                for key, value in payload.items():
                    if isinstance(value, (int, float)):
                        privatized_payload[key] = self._clip_numeric(value) + _laplace(scale)
                    elif isinstance(value, list):
                        privatized_payload[key] = [
                            self._clip_numeric(item) + _laplace(scale) if isinstance(item, (int, float)) else item
                            for item in value
                        ]
                    else:
                        privatized_payload[key] = value
                payload = privatized_payload

            privatized.append(
                KnowledgeArtifact(
                    signature=artifact.signature,
                    text=artifact.text,
                    structured_payload=payload,
                    metadata=metadata,
                )
            )
        if self.adaptive_text_noise:
            return self._apply_adaptive_text_noise(privatized, scale)
        return privatized


    def _clip_numeric(self, value: float) -> float:
        clip_abs = max(float(self.dp_clip_abs), 0.0)
        return max(-clip_abs, min(float(value), clip_abs))

    def _apply_adaptive_text_noise(self, artifacts: Iterable[KnowledgeArtifact], scale: float) -> List[KnowledgeArtifact]:
        processed: List[KnowledgeArtifact] = []
        for artifact in artifacts:
            text = artifact.text or ""
            noisy_text = self._noisify_text(text, scale)
            processed.append(
                KnowledgeArtifact(
                    signature=artifact.signature,
                    text=noisy_text,
                    structured_payload=artifact.structured_payload,
                    metadata=artifact.metadata,
                )
            )
        return processed

    def _noisify_text(self, text: str, scale: float) -> str:
        if not text.strip():
            return text
        tokens = re.split(r"(\s+)", text)
        result: List[str] = []
        for token in tokens:
            if not token or token.isspace():
                result.append(token)
                continue
            saliency = self._token_saliency(token)
            if saliency <= 0:
                result.append(token)
                continue
            probability = min(1.0, saliency * self.adaptive_probability_multiplier)
            if random.random() < probability:
                result.append(self._distort_token(token, scale))
            else:
                result.append(token)
        return "".join(result)

    def _token_saliency(self, token: str) -> float:
        score = 0.0
        if any(ch.isdigit() for ch in token):
            score += self.adaptive_digit_weight
        if len(token) >= 6:
            score += self.adaptive_length_weight
        if token.isupper():
            score += self.adaptive_upper_weight
        if token and token[0].isupper():
            score += self.adaptive_title_weight
        return min(score, 1.0)

    def _distort_token(self, token: str, scale: float) -> str:
        if not token:
            return token
        chars = list(token)
        noise = max(0.1, abs(_laplace(scale))) * max(self.adaptive_distortion_multiplier, 0.1)
        replacements = max(1, min(len(chars), int(round(noise * len(chars)))))
        indices = random.sample(range(len(chars)), replacements)
        for idx in indices:
            chars[idx] = "#"
        return "".join(chars)

    def _looks_like_pii(self, text: str) -> bool:
        """
        Naive PII detector that flags long numeric sequences or email patterns.
        """
        if re.search(r"\b\d{9,}\b", text):
            return True
        if re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text):
            return True
        return False
