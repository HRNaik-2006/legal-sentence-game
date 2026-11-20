"""Offline legal clause classifier with optional local Transformers checkpoint support."""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any

from .core import LEGAL_CLAUSES


def _load_env_file() -> None:
    """Read simple KEY=VALUE settings from the project .env without overriding process env."""
    env_file = Path(".env")
    if not env_file.is_file():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key, value = stripped.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


_load_env_file()
GPT2_MODEL_PATH = os.getenv("GPT2_MODEL_PATH", "./models/legal_gpt2_classifier")

_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Confidentiality": ("confidential", "disclos", "privacy", "safeguard", "secret"),
    "Termination": ("terminat", "terminate", "breach", "cure", "insolvent", "transition"),
    "Indemnification": ("indemn", "defend", "claim", "third party", "infringement"),
    "Governing Law": ("governed", "jurisdiction", "venue", "court", "laws of", "dispute"),
    "Limitation of Liability": ("liabilit", "damages", "liable", "loss", "cap", "exceed"),
    "Force Majeure": ("force majeure", "beyond its control", "event", "suspend", "disaster"),
    "Payment Terms": ("pay", "invoice", "fee", "interest", "amount", "tax", "currency"),
    "Intellectual Property": ("intellectual property", "license", "licence", "ownership", "work product", "proprietary", "rights"),
}

_MODEL: Any = None
_TOKENIZER: Any = None
_MODEL_ATTEMPTED = False


def _local_transformer(text: str) -> dict[str, Any] | None:
    """Use a locally cached fine-tuned model when one is available; never fetch weights."""
    global _MODEL, _TOKENIZER, _MODEL_ATTEMPTED
    model_path = Path(GPT2_MODEL_PATH)
    if not model_path.is_dir():
        return None
    if not _MODEL_ATTEMPTED:
        _MODEL_ATTEMPTED = True
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            _TOKENIZER = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
            _MODEL = AutoModelForSequenceClassification.from_pretrained(str(model_path), local_files_only=True)
            _MODEL.eval()
        except (ImportError, OSError, ValueError):
            _MODEL = None
            _TOKENIZER = None
    if _MODEL is None:
        return None
    import torch
    inputs = _TOKENIZER(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.inference_mode():
        logits = _MODEL(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1)
    index = int(torch.argmax(probs))
    labels = getattr(_MODEL.config, "id2label", {})
    label = str(labels.get(index, labels.get(str(index), "Unknown")))
    return {"clause_type": label, "confidence": float(probs[index]), "feedback": _feedback(label, text)}


def _feedback(clause_type: str, text: str) -> str:
    """Generate concise drafting feedback based on the predicted clause family."""
    suggestions = {
        "Confidentiality": "Define protected information, permitted use, exclusions, and survival period.",
        "Termination": "Specify notice, cure periods, termination triggers, and post-termination duties.",
        "Indemnification": "Clarify covered claims, notice, defense control, and settlement approval.",
        "Governing Law": "Name the governing law and identify an exclusive forum for disputes.",
        "Limitation of Liability": "State the cap, excluded damages, and any exceptions clearly.",
        "Force Majeure": "Define qualifying events, notice, mitigation, and the right to terminate after delay.",
        "Payment Terms": "State invoice timing, payment deadline, disputed amounts, and applicable taxes.",
        "Intellectual Property": "Identify ownership of background materials, deliverables, and license scope.",
    }
    feedback = suggestions.get(clause_type, "State the parties' obligations, timing, and any applicable exceptions.")
    if len(text.split()) < 8:
        feedback += " Add enough detail to make the obligation and its limits clear."
    return feedback


def classify_clause(text: str) -> dict[str, Any]:
    """Classify a clause offline and return clause_type, confidence, and drafting feedback."""
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return {"clause_type": "Unknown", "confidence": 0.0, "feedback": "Enter a clause to receive drafting feedback."}
    model_result = _local_transformer(clean)
    if model_result is not None:
        return model_result
    lowered = clean.lower()
    scores = {label: sum(1.0 for phrase in terms if phrase in lowered) for label, terms in _KEYWORDS.items()}
    # Add a small corpus similarity signal so common contract language is recognized.
    for label, examples in LEGAL_CLAUSES.items():
        for example in examples:
            words = set(re.findall(r"[a-z]+", example.lower()))
            query = set(re.findall(r"[a-z]+", lowered))
            scores[label] += 0.08 * len(words & query) / max(1, len(words))
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    label, best = ranked[0]
    if best <= 0:
        return {"clause_type": "Other", "confidence": 0.25, "feedback": _feedback("Other", clean)}
    denom = sum(value for _, value in ranked)
    confidence = min(0.99, max(0.35, best / denom))
    return {"clause_type": label, "confidence": round(confidence, 4), "feedback": _feedback(label, clean)}


def classify_clause_type(text: str) -> dict[str, Any]:
    """Compatibility alias for the public classifier."""
    return classify_clause(text)
