"""Prometheus metrics for the CUDA server: vLLM's families, names, buckets and labels, with ``tensorfold:`` for ``vllm:``.

``App`` records each request; an engine calls ``round`` once a drafted round is verified and may give ``kv_usage()``.
Everything is host-side arithmetic under one lock: no call here waits on the GPU or changes what is decoded. A family
an engine does not feed is left out of ``/metrics``, never exported as zero.
"""
from __future__ import annotations

import bisect
import threading
import time
from typing import Any

PREFIX = "tensorfold:"

# vLLM's bucket bounds, read from a vLLM server's /metrics
LATENCY = (0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 120.0, 240.0, 480.0,
           960.0, 1920.0, 7680.0)
TTFT = (0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0,
        160.0, 640.0, 2560.0)
ITL = (0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0)
TOKENS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000)

FINISHED_REASONS = ("stop", "length", "abort", "error")      # vLLM's, less "repetition", which nothing here ends on

# name, type, help, buckets; in exposition order
_FAMILIES: tuple[tuple[str, str, str, tuple[float, ...] | None], ...] = (
    ("num_requests_running", "gauge", "Number of requests decoding or prefilling.", None),
    ("num_requests_waiting", "gauge", "Number of requests waiting for the engine.", None),
    ("kv_cache_usage_perc", "gauge", ("KV-cache usage: token positions held by live requests over the positions "
                                      "the cache holds; 1 means 100 percent usage."), None),
    ("prefix_cache_queries", "counter", "Prefix-cache queries, in prompt tokens prefilled.", None),
    ("prefix_cache_hits", "counter", "Prefix-cache hits, in prompt tokens resumed from the cache.", None),
    ("prompt_tokens", "counter", "Number of prefill tokens processed, cache hits included.", None),
    ("prompt_tokens_by_source", "counter", "Number of prompt tokens by source.", None),
    ("prompt_tokens_cached", "counter", "Number of prompt tokens resumed from the prefix cache.", None),
    ("generation_tokens", "counter", "Number of generation tokens processed.", None),
    ("request_success", "counter", "Count of finished requests by finish reason.", None),
    ("spec_decode_num_drafts", "counter", "Number of drafted rounds.", None),
    ("spec_decode_num_draft_tokens", "counter", "Number of draft tokens proposed for verification.", None),
    ("spec_decode_num_accepted_tokens", "counter", "Number of draft tokens accepted.", None),
    ("spec_decode_num_accepted_tokens_per_pos", "counter", "Accepted draft tokens per draft position (0-based).", None),
    ("request_prompt_tokens", "histogram", "Number of prompt tokens per request.", TOKENS),
    ("request_generation_tokens", "histogram", "Number of generation tokens per request.", TOKENS),
    ("request_max_num_generation_tokens", "histogram", "Maximum number of generation tokens per request.", TOKENS),
    ("request_params_max_tokens", "histogram", "The max_tokens request parameter, after the context clamp.", TOKENS),
    ("request_prefill_kv_computed_tokens", "histogram", "Prompt tokens prefilled per request, cache hits excluded.",
     TOKENS),
    ("time_to_first_token_seconds", "histogram", "Time from arrival to the first generated token.", TTFT),
    ("inter_token_latency_seconds", "histogram", "Time between successive token deliveries (one per round).", ITL),
    ("request_time_per_output_token_seconds", "histogram", "Decode time over generation tokens after the first.",
     ITL),
    ("e2e_request_latency_seconds", "histogram", "Time from arrival to the last token.", LATENCY),
    ("request_queue_time_seconds", "histogram", "Time from arrival to the engine taking the request.", LATENCY),
    ("request_inference_time_seconds", "histogram", "Time from the engine taking the request to the last token.",
     LATENCY),
    ("request_prefill_time_seconds", "histogram", "Time from the engine taking the request to the first token.",
     LATENCY),
    ("request_decode_time_seconds", "histogram", "Time from the first token to the last.", LATENCY),
)
_KIND = {name: kind for name, kind, _, _ in _FAMILIES}
_BUCKETS = {name: buckets for name, _, _, buckets in _FAMILIES if buckets}


class Histogram:
    __slots__ = ("bounds", "counts", "sum")

    def __init__(self, bounds: tuple[float, ...]):
        self.bounds, self.counts, self.sum = bounds, [0] * (len(bounds) + 1), 0.0

    def observe(self, value: float) -> None:
        self.counts[bisect.bisect_left(self.bounds, value)] += 1      # le is inclusive
        self.sum += value


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


class Metrics:
    """One served model's families; every method may be called from any thread."""

    def __init__(self, model_name: str):
        self.labels = f'model_name="{_escape(model_name)}"'
        self.lock = threading.Lock()
        self.values: dict[str, dict[str, Any]] = {name: {} for name in _KIND}
        self.running = 0                      # requests between arrival and finish
        self.waiting = 0                      # of those, requests waiting for the serialized engine
        for name in ("prefix_cache_queries", "prefix_cache_hits", "prompt_tokens", "prompt_tokens_cached",
                     "generation_tokens"):
            self.values[name][""] = 0
        for source in ("local_compute", "local_cache_hit"):
            self.values["prompt_tokens_by_source"][f',source="{source}"'] = 0
        for reason in FINISHED_REASONS:
            self.values["request_success"][f',finished_reason="{reason}"'] = 0
        for name, bounds in _BUCKETS.items():
            self.values[name][""] = Histogram(bounds)

    def add(self, name: str, amount: float, labels: str = "") -> None:
        with self.lock:
            self.values[name][labels] = self.values[name].get(labels, 0) + amount

    def observe(self, name: str, value: float) -> None:
        with self.lock:
            self.values[name][""].observe(value)

    def round(self, drafted: int, accepted: int, positions: int | None = None) -> None:
        if drafted <= 0:
            return
        per_pos = self.values["spec_decode_num_accepted_tokens_per_pos"]
        with self.lock:
            for name, amount in (("spec_decode_num_drafts", 1), ("spec_decode_num_draft_tokens", drafted),
                                 ("spec_decode_num_accepted_tokens", accepted)):
                self.values[name][""] = self.values[name].get("", 0) + amount
            for position in range(max(positions or drafted, accepted)):
                key = f',position="{position}"'
                per_pos[key] = per_pos.get(key, 0) + (position < accepted)

    def render(self, engine: Any = None) -> str:
        """The exposition, Prometheus text format 0.0.4; ``engine`` gives the queue and KV gauges read now."""

        queued = getattr(getattr(engine, "scheduler", None), "waiting", None)   # a ``--parallel`` engine's queue
        queued = queued.qsize() if queued is not None else 0
        usage = getattr(engine, "kv_usage", None)
        usage = usage() if usage is not None else None
        lines = []
        with self.lock:
            running = max(0, self.running - self.waiting - queued)
            gauges = {"num_requests_waiting": self.waiting + queued, "num_requests_running": running}
            if usage is not None and usage[1] > 0:
                # with nothing running, a serialized engine's state is a finished reply's: kept for reuse, not in use
                gauges["kv_cache_usage_perc"] = min(1.0, usage[0] / usage[1]) if running else 0.0
            for name, kind, text, _ in _FAMILIES:
                series = {"": gauges[name]} if name in gauges else self.values[name]
                if not series:
                    continue
                full = PREFIX + name
                lines += [f"# HELP {full} {text}", f"# TYPE {full} {kind}"]
                for labels, value in series.items():
                    if kind == "histogram":
                        total = 0
                        for bound, count in zip((*value.bounds, float("inf")), value.counts):
                            total += count
                            le = "+Inf" if bound == float("inf") else repr(float(bound))
                            lines.append(f'{full}_bucket{{{self.labels}{labels},le="{le}"}} {total}')
                        lines.append(f"{full}_sum{{{self.labels}{labels}}} {value.sum!r}")
                        lines.append(f"{full}_count{{{self.labels}{labels}}} {total}")
                    else:
                        suffix = "_total" if kind == "counter" else ""
                        lines.append(f"{full}{suffix}{{{self.labels}{labels}}} {float(value)!r}")
        return "\n".join(lines) + "\n"


_current: Metrics | None = None


def install(registry: Metrics | None) -> None:
    """The registry engine rounds report to; only the process serving HTTP has one (rank 1 never does)."""

    global _current
    _current = registry


def round(drafted: int, accepted: int, positions: int | None = None) -> None:
    """A verified round: ``drafted`` tokens proposed, the first ``accepted`` of its deepest path kept.

    ``positions`` is how deep the draft went (a tree's depth; a chain's length, the default)."""

    registry = _current
    if registry is not None:
        registry.round(drafted, accepted, positions)


class RequestClock:
    """One request's times and token counts, reported when it finishes; with no registry every call does nothing."""

    def __init__(self, registry: Metrics | None, arrived: float | None, prompt_tokens: int, max_tokens: int):
        self.m = registry
        self.arrived = arrived if arrived is not None else time.perf_counter()
        self.prompt_tokens, self.max_tokens = prompt_tokens, max_tokens
        self.scheduled_at: float | None = None
        self.first: float | None = None
        self.last: float | None = None
        self.generated = self.prefilled = 0
        self.stats: dict[str, Any] | None = None
        self.in_queue = False
        if self.m is not None:
            with self.m.lock:
                self.m.running += 1

    def queued(self) -> None:
        """The request waits for the serialized engine."""

        if self.m is not None:
            with self.m.lock:
                self.m.waiting += 1
            self.in_queue = True

    def scheduled(self) -> None:
        self._dequeue()
        self.scheduled_at = time.perf_counter()

    def prefill(self, tokens: int) -> None:
        """One call into the engine with this many prompt tokens (a tool-call gate's continuation calls again)."""

        self.prefilled += tokens

    def tokens(self, count: int) -> None:
        """A round's tokens reached the request."""

        if self.m is None or not count:
            return
        now = time.perf_counter()
        self.m.observe(*(("time_to_first_token_seconds", now - self.arrived) if self.first is None
                         else ("inter_token_latency_seconds", now - self.last)))
        self.first = now if self.first is None else self.first
        self.last = now
        self.generated += count
        self.m.add("generation_tokens", count)

    def engine_returned(self, stats: dict[str, Any] | None) -> None:
        self.stats = dict(stats or {})

    def finish(self, reason: str) -> None:
        if self.m is None:
            return
        self._dequeue()
        end = time.perf_counter()
        m = self.m
        with m.lock:
            m.running -= 1
        m.add("request_success", 1, f',finished_reason="{reason}"')
        if self.stats is None:                  # the engine never returned: its prompt counts are unknown
            return
        cached = min(int(self.stats.get("cached") or 0), self.prefilled)
        computed = self.prefilled - cached
        m.add("prompt_tokens", self.prefilled)
        m.add("prompt_tokens_by_source", computed, ',source="local_compute"')
        m.add("prompt_tokens_by_source", cached, ',source="local_cache_hit"')
        m.add("prompt_tokens_cached", cached)
        m.add("prefix_cache_queries", self.prefilled)
        m.add("prefix_cache_hits", cached)
        # a ``--parallel`` engine queues requests itself and reports how long this one waited
        scheduled = (self.scheduled_at or self.arrived) + float(self.stats.get("queue_s") or 0.0)
        first, last = self.first, self.last if self.last is not None else end
        for name, value in (("request_prompt_tokens", self.prompt_tokens),
                            ("request_generation_tokens", self.generated),
                            ("request_max_num_generation_tokens", self.generated),
                            ("request_params_max_tokens", self.max_tokens),
                            ("request_prefill_kv_computed_tokens", computed),
                            ("e2e_request_latency_seconds", last - self.arrived),
                            ("request_queue_time_seconds", scheduled - self.arrived),
                            ("request_inference_time_seconds", last - scheduled)):
            m.observe(name, max(0.0, value))
        if first is not None:
            m.observe("request_prefill_time_seconds", max(0.0, first - scheduled))
            m.observe("request_decode_time_seconds", max(0.0, last - first))
            if self.generated > 1:
                m.observe("request_time_per_output_token_seconds", (last - first) / (self.generated - 1))

    def _dequeue(self) -> None:
        if self.in_queue:
            self.in_queue = False
            with self.m.lock:
                self.m.waiting -= 1
