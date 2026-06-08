"""Session-scoped execution telemetry for LLM routing decisions.

Tracks per-model execution stats (latency, reliability, cost, quality)
using a sliding-window approach to bound memory usage.  Thread-safe.

Usage::

    tel = RoutingTelemetry()
    tel.record("mistral:7b", "chat", latency_ms=320, input_tokens=128,
               output_tokens=64, success=True, quality_score=0.9)
    stats = tel.get_model_stats("mistral:7b")
"""

from __future__ import annotations

import json
import logging
import math
import threading
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_WINDOW_SIZE = 100

# ── Relative cost weights per token (unitless, for comparison only) ──────────
# Local models are effectively free; cloud models vary.  These are rough
# multipliers applied to (input_tokens + output_tokens) to produce a
# comparable "cost units" number.  Extend as needed.
_DEFAULT_COST_WEIGHTS: dict[str, float] = {
    # Local / Ollama — near-zero real cost
    "llama3.2:1b":      0.01,
    "qwen2.5:0.5b":     0.01,
    "qwen2.5:1.5b":     0.01,
    "gemma2:2b":         0.01,
    "mistral:7b":        0.02,
    "llama3:8b":         0.02,
    "qwen2.5:7b":        0.02,
    "deepseek-r1:8b":    0.03,
    "deepseek-r1:14b":   0.05,
    "llama3.3:70b":      0.10,
    # Cloud models — higher weight
    "gemini":            0.50,
    "groq":              0.30,
    "openai":            1.00,
    "anthropic":         1.00,
}

_DEFAULT_WEIGHT = 0.05  # fallback for unknown models


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass(slots=True)
class _CallRecord:
    """Single LLM call record kept in the sliding window."""

    model: str
    task_type: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    success: bool
    quality_score: float | None = None


@dataclass(slots=True)
class ModelStats:
    """Aggregate statistics for a single model."""

    model: str
    total_calls: int = 0
    success_count: int = 0
    failure_count: int = 0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    avg_quality: float = 0.5
    success_rate: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0


# ── Core tracker ─────────────────────────────────────────────────────────────

class RoutingTelemetry:
    """Session-scoped telemetry tracker for LLM routing decisions.

    Keeps the last ``_WINDOW_SIZE`` records per model to cap memory.
    All public methods are thread-safe.
    """

    def __init__(
        self,
        *,
        cost_weights: dict[str, float] | None = None,
        window_size: int = _WINDOW_SIZE,
    ) -> None:
        self._lock = threading.Lock()
        self._windows: dict[str, deque[_CallRecord]] = {}
        self._window_size = window_size
        self._cost_weights = cost_weights or _DEFAULT_COST_WEIGHTS

    # ── Recording ────────────────────────────────────────────────────────

    def record(
        self,
        model: str,
        task_type: str,
        latency_ms: float,
        input_tokens: int,
        output_tokens: int,
        success: bool,
        quality_score: float | None = None,
    ) -> None:
        """Record a single LLM call result."""
        rec = _CallRecord(
            model=model,
            task_type=task_type,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            success=success,
            quality_score=quality_score,
        )
        with self._lock:
            window = self._windows.setdefault(
                model, deque(maxlen=self._window_size),
            )
            window.append(rec)

    # ── Query helpers ────────────────────────────────────────────────────

    def get_model_stats(self, model: str) -> ModelStats:
        """Return aggregate stats for *model* across its sliding window."""
        with self._lock:
            records = list(self._windows.get(model, []))

        if not records:
            return ModelStats(model=model)

        total = len(records)
        successes = sum(1 for r in records if r.success)
        failures = total - successes

        latencies = [r.latency_ms for r in records]
        avg_lat = sum(latencies) / total
        p95_lat = _percentile(latencies, 95)

        quality_vals = [r.quality_score for r in records if r.quality_score is not None]
        avg_q = sum(quality_vals) / len(quality_vals) if quality_vals else 0.5

        return ModelStats(
            model=model,
            total_calls=total,
            success_count=successes,
            failure_count=failures,
            avg_latency_ms=round(avg_lat, 2),
            p95_latency_ms=round(p95_lat, 2),
            avg_quality=round(avg_q, 4),
            success_rate=round(successes / total, 4),
            total_input_tokens=sum(r.input_tokens for r in records),
            total_output_tokens=sum(r.output_tokens for r in records),
        )

    def get_reliability(self, model: str, task_type: str) -> float:
        """Return 0.0–1.0 reliability for *model* on a specific *task_type*."""
        with self._lock:
            records = [
                r for r in self._windows.get(model, [])
                if r.task_type == task_type
            ]
        if not records:
            return 0.0
        return round(sum(1 for r in records if r.success) / len(records), 4)

    def get_avg_latency(self, model: str) -> float:
        """Return average latency in ms across the sliding window."""
        with self._lock:
            records = list(self._windows.get(model, []))
        if not records:
            return 0.0
        return round(sum(r.latency_ms for r in records) / len(records), 2)

    def get_cost_estimate(self, model: str) -> float:
        """Return estimated relative cost based on recorded tokens × model weight."""
        with self._lock:
            records = list(self._windows.get(model, []))
        if not records:
            return 0.0
        weight = self._resolve_cost_weight(model)
        total_tokens = sum(r.input_tokens + r.output_tokens for r in records)
        return round(total_tokens * weight, 4)

    def summary(self) -> dict[str, Any]:
        """Full stats summary for every tracked model."""
        with self._lock:
            models = list(self._windows.keys())
        return {m: asdict(self.get_model_stats(m)) for m in models}

    # ── Persistence ──────────────────────────────────────────────────────

    def save_to_file(self, path: str | Path) -> None:
        """Persist all records as JSONL (one JSON object per line)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            all_records = [
                rec
                for window in self._windows.values()
                for rec in window
            ]
        try:
            with path.open("w", encoding="utf-8") as fh:
                for rec in all_records:
                    line = json.dumps({
                        "model": rec.model,
                        "task_type": rec.task_type,
                        "latency_ms": rec.latency_ms,
                        "input_tokens": rec.input_tokens,
                        "output_tokens": rec.output_tokens,
                        "success": rec.success,
                        "quality_score": rec.quality_score,
                    }, ensure_ascii=False)
                    fh.write(line + "\n")
            logger.debug("Saved %d telemetry records to %s", len(all_records), path)
        except OSError as exc:
            logger.warning("Failed to save telemetry to %s: %s", path, exc)

    @classmethod
    def load_from_file(cls, path: str | Path, **kwargs: Any) -> RoutingTelemetry:
        """Load telemetry from a JSONL file, returning a new instance.

        Extra *kwargs* are forwarded to the ``RoutingTelemetry`` constructor.
        """
        instance = cls(**kwargs)
        path = Path(path)
        if not path.is_file():
            logger.debug("Telemetry file %s not found — starting fresh", path)
            return instance

        loaded = 0
        try:
            with path.open("r", encoding="utf-8") as fh:
                for lineno, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        instance.record(
                            model=str(obj["model"]),
                            task_type=str(obj["task_type"]),
                            latency_ms=float(obj["latency_ms"]),
                            input_tokens=int(obj["input_tokens"]),
                            output_tokens=int(obj["output_tokens"]),
                            success=bool(obj["success"]),
                            quality_score=(
                                float(obj["quality_score"])
                                if obj.get("quality_score") is not None
                                else None
                            ),
                        )
                        loaded += 1
                    except (KeyError, ValueError, TypeError) as exc:
                        logger.warning(
                            "Skipping malformed telemetry line %d: %s", lineno, exc,
                        )
        except OSError as exc:
            logger.warning("Failed to read telemetry from %s: %s", path, exc)

        logger.debug("Loaded %d telemetry records from %s", loaded, path)
        return instance

    # ── Internal ─────────────────────────────────────────────────────────

    def _resolve_cost_weight(self, model: str) -> float:
        """Look up cost weight, falling back to family prefix matching."""
        if model in self._cost_weights:
            return self._cost_weights[model]
        # Try family prefix (e.g. "mistral:7b" → "mistral")
        family = model.split(":")[0]
        if family in self._cost_weights:
            return self._cost_weights[family]
        return _DEFAULT_WEIGHT


# ── Utilities ────────────────────────────────────────────────────────────────

def _percentile(values: list[float], pct: int) -> float:
    """Compute the *pct*-th percentile of *values* (nearest-rank method)."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = max(0, math.ceil(len(sorted_vals) * pct / 100) - 1)
    return sorted_vals[idx]


__all__ = ["RoutingTelemetry", "ModelStats"]
