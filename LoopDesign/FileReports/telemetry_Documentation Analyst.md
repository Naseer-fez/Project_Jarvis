# Analysis Report for telemetry.py

## Dependencies
- __future__.annotations
- json
- logging
- math
- threading
- collections.deque
- dataclasses.asdict
- dataclasses.dataclass
- pathlib.Path
- typing.Any

## Schemas
- _CallRecord
- _CallRecord attribute: model
- _CallRecord attribute: task_type
- _CallRecord attribute: latency_ms
- _CallRecord attribute: input_tokens
- _CallRecord attribute: output_tokens
- _CallRecord attribute: success
- _CallRecord attribute: quality_score
- ModelStats
- ModelStats attribute: model
- ModelStats attribute: total_calls
- ModelStats attribute: success_count
- ModelStats attribute: failure_count
- ModelStats attribute: avg_latency_ms
- ModelStats attribute: p95_latency_ms
- ModelStats attribute: avg_quality
- ModelStats attribute: success_rate
- ModelStats attribute: total_input_tokens
- ModelStats attribute: total_output_tokens
- RoutingTelemetry

## API Contracts
- RoutingTelemetry.__init__(self)
- RoutingTelemetry.record(self, model, task_type, latency_ms, input_tokens, output_tokens, success, quality_score)
- RoutingTelemetry.get_model_stats(self, model)
- RoutingTelemetry.get_reliability(self, model, task_type)
- RoutingTelemetry.get_avg_latency(self, model)
- RoutingTelemetry.get_cost_estimate(self, model)
- RoutingTelemetry.summary(self)
- RoutingTelemetry.save_to_file(self, path)
- RoutingTelemetry.load_from_file(cls, path)
- RoutingTelemetry._resolve_cost_weight(self, model)
- _percentile(values, pct)

## Configuration Variables
- _WINDOW_SIZE
- _DEFAULT_COST_WEIGHTS (typed)
- _DEFAULT_WEIGHT

## Assumptions & Notes
- Module Docstring: Session-scoped execution telemetry for LLM routing decisions.

Tracks per-model execution stats (latency, reliability, cost, quality)
using a sliding-window approach to bound memory usage.  Thread-safe.

Usage::

    tel = RoutingTelemetry()
    tel.record("mistral:7b", "chat", latency_ms=320, input_tokens=128,
               output_tokens=64, success=True, quality_score=0.9)
    stats = tel.get_model_stats("mistral:7b")

