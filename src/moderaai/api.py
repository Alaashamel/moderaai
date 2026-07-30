from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .classifier import LexicalClassifier

app = FastAPI(
    title="moderaai",
    description="Lightweight text moderation API — classifies content as safe, toxic, or spam.",
    version="0.1.0",
)

_classifier: LexicalClassifier | None = None


def get_classifier() -> LexicalClassifier:
    global _classifier
    if _classifier is None:
        _classifier = LexicalClassifier.load_or_train()
    return _classifier


class ModerateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)


class ModerateResponse(BaseModel):
    label: str
    is_safe: bool
    scores: dict[str, float]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/moderate/text", response_model=ModerateResponse)
def moderate_text(req: ModerateRequest):
    result = get_classifier().predict(req.text)
    return ModerateResponse(label=result.label, is_safe=result.is_safe, scores=result.scores)
