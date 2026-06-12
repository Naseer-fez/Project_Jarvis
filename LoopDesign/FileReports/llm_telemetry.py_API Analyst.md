# API Analyst Report: llm\telemetry.py

## Dependencies
- `from __future__ import annotations`
- `import json`
- `import logging`
- `import math`
- `import threading`
- `from collections import deque`
- `from dataclasses import asdict`
- `from dataclasses import dataclass`
- `from pathlib import Path`
- `from typing import Any`

## Configuration Variables
- `_WINDOW_SIZE` = `100`
- `_DEFAULT_WEIGHT` = `0.05`

## Schemas & API Contracts (Classes)

### Class `_CallRecord`
> Single LLM call record kept in the sliding window.

**Fields/Schema:**
  - `model: str`
  - `task_type: str`
  - `latency_ms: float`
  - `input_tokens: int`
  - `output_tokens: int`
  - `success: bool`
  - `quality_score: float | None`



### Class `ModelStats`
> Aggregate statistics for a single model.

**Fields/Schema:**
  - `model: str`
  - `total_calls: int`
  - `success_count: int`
  - `failure_count: int`
  - `avg_latency_ms: float`
  - `p95_latency_ms: float`
  - `avg_quality: float`
  - `success_rate: float`
  - `total_input_tokens: int`
  - `total_output_tokens: int`



### Class `RoutingTelemetry`
> Session-scoped telemetry tracker for LLM routing decisions.

Keeps the last ``_WINDOW_SIZE`` records per model to cap memory.
All public methods are thread-safe.

**Methods:**
- `def __init__(self, *, cost_weights: dict[str, float] | None=None, window_size: int=_WINDOW_SIZE) -> None`
- `def record(self, model: str, task_type: str, latency_ms: float, input_tokens: int, output_tokens: int, success: bool, quality_score: float | None=None) -> None`
  - *Record a single LLM call result.*
- `def get_model_stats(self, model: str) -> ModelStats`
  - *Return aggregate stats for *model* across its sliding window.*
- `def get_reliability(self, model: str, task_type: str) -> float`
  - *Return 0.0–1.0 reliability for *model* on a specific *task_type*.*
- `def get_avg_latency(self, model: str) -> float`
  - *Return average latency in ms across the sliding window.*
- `def get_cost_estimate(self, model: str) -> float`
  - *Return estimated relative cost based on recorded tokens × model weight.*
- `def summary(self) -> dict[str, Any]`
  - *Full stats summary for every tracked model.*
- `def save_to_file(self, path: str | Path) -> None`
  - *Persist all records as JSONL (one JSON object per line).*
- @classmethod
- `def load_from_file(cls, path: str | Path, **kwargs: Any) -> RoutingTelemetry`
  - *Load telemetry from a JSONL file, returning a new instance.*
- `def _resolve_cost_weight(self, model: str) -> float`
  - *Look up cost weight, falling back to family prefix matching.*


## Functions & Endpoints

### `_percentile`
`def _percentile(values: list[float], pct: int) -> float`
> Compute the *pct*-th percentile of *values* (nearest-rank method).
