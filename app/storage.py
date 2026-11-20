"""JSON-backed local round history and progress tracking."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any


def _read_env_setting(key: str, default: str) -> str:
    """Read a simple project .env value without overriding process environment."""
    if key in os.environ:
        return os.environ[key]
    env_file = Path(".env")
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                name, value = stripped.split("=", 1)
                if name.strip() == key:
                    return value.strip().strip("\"'")
    return default


DEFAULT_STORAGE_PATH = Path(_read_env_setting("LEGAL_GAME_DATA", ".data/history.json"))


def _read(path: str | Path | None) -> dict[str, Any]:
    target = Path(path) if path is not None else DEFAULT_STORAGE_PATH
    if not target.exists():
        return {"rounds": []}
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("rounds"), list):
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return {"rounds": []}


def load_history(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load all saved rounds, returning an empty list when no history exists."""
    return list(_read(path)["rounds"])


def save_round(round_data: dict[str, Any], path: str | Path | None = None) -> dict[str, Any]:
    """Append one round to local history, creating its parent directory as needed."""
    target = Path(path) if path is not None else DEFAULT_STORAGE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    data = _read(target)
    record = dict(round_data)
    record.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    data["rounds"].append(record)
    target.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return record


def record_round(target_type: str, predicted_type: str, sentence: str, confidence: float, feedback: str = "", path: str | Path | None = None) -> dict[str, Any]:
    """Record a completed round and derive target-match progress fields."""
    return save_round({"target_type": target_type, "predicted_type": predicted_type,
                       "sentence": sentence, "confidence": float(confidence),
                       "feedback": feedback, "matched": target_type == predicted_type}, path)


def get_progress(path: str | Path | None = None) -> dict[str, int | float]:
    """Summarize played rounds, target matches, and match rate."""
    rounds = load_history(path)
    matches = sum(bool(item.get("matched")) for item in rounds)
    return {"rounds_played": len(rounds), "matches": matches,
            "match_rate": matches / len(rounds) if rounds else 0.0}


def clear_history(path: str | Path | None = None) -> None:
    """Remove saved progress at the requested path (primarily useful for isolated tests)."""
    target = Path(path) if path is not None else DEFAULT_STORAGE_PATH
    if target.exists():
        target.unlink()
