"""Text moderation classifier.

Reference implementation: TF-IDF + multinomial logistic regression,
trained on a small bundled labeled dataset (`data/labeled_examples.csv`).
This is intentionally lightweight — no GPU, no multi-hundred-megabyte
download, trains in well under a second, and ships as a fully reproducible
artifact you can inspect line by line.

This is a *reference* implementation, not a production-accuracy one: the
bundled dataset is small (60 examples), so treat scores as indicative
rather than authoritative. For production-grade accuracy, swap in a
transformer model (see `TransformerClassifier` below and the README) —
the `Classifier` protocol makes that a drop-in replacement.
"""

from __future__ import annotations

import csv
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

DATA_PATH = Path(__file__).parent / "data" / "labeled_examples.csv"
DEFAULT_MODEL_PATH = Path(__file__).parent / "data" / "model.pkl"

LABELS = ("safe", "toxic", "spam")


@dataclass
class ModerationResult:
    label: str
    scores: dict[str, float]

    @property
    def is_safe(self) -> bool:
        return self.label == "safe"


class Classifier(Protocol):
    def predict(self, text: str) -> ModerationResult: ...


def load_training_data(path: Path = DATA_PATH) -> tuple[list[str], list[str]]:
    texts, labels = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            texts.append(row["text"])
            labels.append(row["label"])
    return texts, labels


def train_model(data_path: Path = DATA_PATH) -> Pipeline:
    texts, labels = load_training_data(data_path)
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, lowercase=True)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    pipeline.fit(texts, labels)
    return pipeline


class LexicalClassifier:
    """The default, always-available classifier: TF-IDF + logistic regression."""

    def __init__(self, model: Pipeline | None = None):
        self._model = model or train_model()

    @classmethod
    def load_or_train(cls, model_path: Path = DEFAULT_MODEL_PATH) -> "LexicalClassifier":
        if model_path.exists():
            with open(model_path, "rb") as f:
                return cls(model=pickle.load(f))
        instance = cls()
        instance.save(model_path)
        return instance

    def save(self, path: Path = DEFAULT_MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._model, f)

    def predict(self, text: str) -> ModerationResult:
        proba = self._model.predict_proba([text])[0]
        classes = list(self._model.classes_)
        scores = {label: float(proba[classes.index(label)]) if label in classes else 0.0 for label in LABELS}
        label = max(scores, key=scores.get)
        return ModerationResult(label=label, scores=scores)


class TransformerClassifier:
    """Optional production-grade backend using a Hugging Face model
    (e.g. `unitary/toxic-bert`). Requires the `transformers` + `torch`
    extras (`pip install moderaai[transformer]`). Not used by default —
    see README for when to reach for this instead of `LexicalClassifier`.
    """

    def __init__(self, model_name: str = "unitary/toxic-bert"):
        try:
            from transformers import pipeline
        except ImportError as e:
            raise ImportError(
                "TransformerClassifier requires the 'transformer' extra: "
                "pip install moderaai[transformer]"
            ) from e
        self._pipe = pipeline("text-classification", model=model_name, top_k=None)

    def predict(self, text: str) -> ModerationResult:
        raw = self._pipe(text)[0]
        scores = {item["label"].lower(): float(item["score"]) for item in raw}
        # Normalize toxic-bert's binary output onto our (safe, toxic, spam) schema.
        toxic_score = scores.get("toxic", 0.0)
        normalized = {"safe": 1 - toxic_score, "toxic": toxic_score, "spam": 0.0}
        label = "toxic" if toxic_score >= 0.5 else "safe"
        return ModerationResult(label=label, scores=normalized)
