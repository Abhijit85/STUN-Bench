from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from openrouter_client import chat_completion
from synapse.clients.client import SynapseClient

try:  # pragma: no cover - optional dependency
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:  # pragma: no cover
    TfidfVectorizer = None

try:  # pragma: no cover - optional dependency
    from rouge_score import rouge_scorer
except ImportError:  # pragma: no cover
    rouge_scorer = None

_ROUGE_SCORER = (
    rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True) if rouge_scorer else None
)

ATTACK_SYSTEM_PROMPT = (
    "You are an adversarial analyst attempting to reconstruct the original "
    "prompt or in-context example that produced a public response. "
    "Given only the observed response, infer the most likely original prompt. "
    "Respond with your best guess as plain text without extra commentary."
)


@dataclass
class AttackSample:
    client_id: str
    signature: str
    raw_text: str
    observed_text: str
    reconstructed_text: Optional[str] = None
    similarity: Optional[float] = None
    cosine_similarity: Optional[float] = None
    rouge_l: Optional[float] = None


def _tokenize(text: str) -> List[str]:
    return [token for token in text.lower().split() if token]


def _similarity_score(reference: str, guess: str) -> float:
    """
    Simple recall-style overlap score between the original prompt and reconstruction.
    """
    ref_tokens = _tokenize(reference)
    if not ref_tokens:
        return 0.0
    guess_tokens = set(_tokenize(guess))
    overlap = sum(1 for token in ref_tokens if token in guess_tokens)
    return overlap / len(ref_tokens)


def _cosine_similarity(text_a: str, text_b: str) -> Optional[float]:
    if TfidfVectorizer is None:
        return None
    texts = [text_a or "", text_b or ""]
    if not any(t.strip() for t in texts):
        return None
    try:
        vectorizer = TfidfVectorizer().fit(texts)
        vectors = vectorizer.transform(texts).toarray()
    except ValueError:  # empty vocabulary
        return None
    vec_a, vec_b = vectors
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denom == 0:
        return None
    return float(np.dot(vec_a, vec_b) / denom)


def _rouge_l_score(reference: str, guess: str) -> Optional[float]:
    if _ROUGE_SCORER is None:
        return None
    if not reference.strip() or not guess.strip():
        return None
    scores = _ROUGE_SCORER.score(reference, guess)
    return getattr(scores["rougeL"], "f" + "me" + chr(97) + "sure")


def _reconstruct_prompt(observed_text: str, *, model: str) -> str:
    messages = [
        {"role": "system", "content": ATTACK_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Observed response from the client:\n"
                f"{observed_text}\n\n"
                "Reconstruct the original prompt/in-context example."
            ),
        },
    ]
    completion = chat_completion(model=model, messages=messages, max_tokens=256)
    return completion.choices[0].message.content.strip()


def _collect_attack_samples(clients: Sequence[SynapseClient], limit: int) -> List[AttackSample]:
    samples: List[AttackSample] = []
    for client in clients:
        artifacts = client.get_attack_artifacts()
        if not artifacts:
            continue
        for entry in artifacts[:limit]:
            samples.append(
                AttackSample(
                    client_id=client.metadata.client_id,
                    signature=entry["signature"],
                    raw_text=entry["raw_text"],
                    observed_text=entry["observed_text"],
                )
            )
    return samples


def run_prompt_extraction_attack(
    clients: Sequence[SynapseClient],
    *,
    model: str,
    max_samples: int,
    output_path: Optional[Path] = None,
) -> Dict[str, object]:
    """
    Attempt to reconstruct client prompts using only sanitized artifacts.
    Returns summary metrics and optionally writes the detailed report.
    """
    samples = _collect_attack_samples(clients, max_samples)
    if not samples:
        return {
            "samples": 0,
            "average_similarity": None,
            "average_cosine_similarity": None,
            "average_rouge_l": None,
            "details": [],
        }

    for sample in samples:
        try:
            reconstruction = _reconstruct_prompt(sample.observed_text, model=model)
        except Exception as exc:  # pragma: no cover - network
            reconstruction = f"[attack_failed: {exc}]"
        sample.reconstructed_text = reconstruction
        sample.similarity = _similarity_score(sample.raw_text, reconstruction)
        sample.cosine_similarity = _cosine_similarity(sample.raw_text, reconstruction)
        sample.rouge_l = _rouge_l_score(sample.raw_text, reconstruction)

    avg_similarity = sum(s.similarity or 0.0 for s in samples) / len(samples)
    cosine_values = [s.cosine_similarity for s in samples if s.cosine_similarity is not None]
    rouge_values = [s.rouge_l for s in samples if s.rouge_l is not None]
    avg_cosine = sum(cosine_values) / len(cosine_values) if cosine_values else None
    avg_rouge = sum(rouge_values) / len(rouge_values) if rouge_values else None
    details = [
        {
            "client_id": s.client_id,
            "signature": s.signature,
            "token_overlap": s.similarity,
            "cosine_similarity": s.cosine_similarity,
            "rouge_l": s.rouge_l,
            "original_prompt": s.raw_text,
            "observed_response": s.observed_text,
            "reconstructed_prompt": s.reconstructed_text,
        }
        for s in samples
    ]

    report = {
        "samples": len(samples),
        "average_similarity": avg_similarity,
        "average_cosine_similarity": avg_cosine,
        "average_rouge_l": avg_rouge,
        "model": model,
        "details": details,
    }

    if output_path:
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report
