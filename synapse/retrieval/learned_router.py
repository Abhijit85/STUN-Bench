from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


DEFAULT_LABEL_REMAP = {
    "Number Theory": "General Logic and Counting",
}

DEFAULT_DROPPED_LABELS = {"SearchQA"}


@dataclass
class RoutingExample:
    query_id: str
    sample_id: str
    query_text: str
    label: str


def load_routing_examples(
    path: Path,
    *,
    label_remap: dict[str, str] | None = None,
    dropped_labels: set[str] | None = None,
) -> list[RoutingExample]:
    remap = label_remap or DEFAULT_LABEL_REMAP
    dropped = dropped_labels or DEFAULT_DROPPED_LABELS
    examples: list[RoutingExample] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            router = row.get("router") or {}
            label = router.get("ground_truth_domain")
            query_text = row.get("query_text")
            if not isinstance(label, str) or not label.strip():
                continue
            if not isinstance(query_text, str) or not query_text.strip():
                continue
            label = remap.get(label.strip(), label.strip())
            if label in dropped:
                continue
            examples.append(
                RoutingExample(
                    query_id=str(row.get("query_id") or ""),
                    sample_id=str(row.get("sample_id") or ""),
                    query_text=query_text.strip(),
                    label=label,
                )
            )
    return examples


def build_default_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        ("word", TfidfVectorizer(ngram_range=(1, 2), stop_words="english")),
                        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))),
                    ]
                ),
            ),
            ("classifier", LinearSVC()),
        ]
    )


def _safe_n_splits(labels: list[str], requested: int) -> int:
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    min_count = min(counts.values()) if counts else 0
    return max(2, min(requested, min_count))


class LearnedTextRouter:
    def __init__(self, pipeline: Pipeline | None = None) -> None:
        self.pipeline = pipeline or build_default_pipeline()

    def fit(self, examples: Iterable[RoutingExample]) -> "LearnedTextRouter":
        rows = list(examples)
        self.pipeline.fit([row.query_text for row in rows], [row.label for row in rows])
        return self

    def predict(self, texts: Iterable[str]) -> list[str]:
        return [str(value) for value in self.pipeline.predict(list(texts))]

    def predict_one(self, text: str) -> str:
        return self.predict([text])[0]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path: Path) -> "LearnedTextRouter":
        return cls(pipeline=joblib.load(path))


def cross_validated_predictions(
    examples: list[RoutingExample],
    *,
    n_splits: int = 5,
    random_state: int = 0,
) -> list[dict[str, Any]]:
    texts = [row.query_text for row in examples]
    labels = [row.label for row in examples]
    splits = _safe_n_splits(labels, n_splits)
    splitter = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)
    rows: list[dict[str, Any] | None] = [None] * len(examples)

    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(texts, labels), start=1):
        router = LearnedTextRouter().fit(examples[idx] for idx in train_idx)
        predictions = router.predict(texts[idx] for idx in test_idx)
        for idx, prediction in zip(test_idx, predictions):
            example = examples[idx]
            rows[idx] = {
                "query_id": example.query_id,
                "sample_id": example.sample_id,
                "query_text": example.query_text,
                "ground_truth_domain": example.label,
                "predicted_domain": prediction,
                "routed_correctly": prediction == example.label,
                "fold": fold_id,
            }

    return [row for row in rows if row is not None]
