"""The CUDA server's ``/metrics``: what ``App.run`` records for each request, on both request paths."""

import http.client
import queue
import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("jinja2")

from test_cuda_metrics import samples
from test_cuda_server_disconnect import MESSAGES, WAIT, PacedEngine, SchedulerEngine, app_for, serving, until

from tensorfold.cuda import metrics
from tensorfold.cuda.metrics import Metrics
from tensorfold.server.cancellation import RequestCancelled

M = '{model_name="fake-cuda"}'


class CachedEngine(PacedEngine):
    """A paced engine that resumed ``cached`` prompt tokens from its prefix cache."""

    def __init__(self, text="", *, cached=0, **kwargs):
        super().__init__(text, **kwargs)
        self.cached = cached

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
        return {**super().generate(prompt, max_tokens, sampling, on_tokens, draft), "cached": self.cached}


class FailingEngine(PacedEngine):
    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
        on_tokens([ord("x")])
        raise RuntimeError("the engine failed")


def measured(tmp_path, engine):
    app = app_for(tmp_path, engine)
    app.metrics = Metrics(app.served)
    return app


def scrape(app) -> dict[str, float]:
    return samples(app.metrics.render(app.engine))


def success(s, reason) -> float:
    return s[f'tensorfold:request_success_total{{model_name="fake-cuda",finished_reason="{reason}"}}']


@pytest.fixture(autouse=True)
def no_sink():
    yield
    metrics.install(None)


def test_a_finished_request_counts_its_tokens_and_times(tmp_path):
    app = measured(tmp_path, CachedEngine(cached=3))
    result = app.run({"messages": MESSAGES, "max_tokens": 5}, True, lambda delta: True)
    prompt = result["prompt_tokens"]
    s = scrape(app)
    assert success(s, "length") == 1
    assert s[f"tensorfold:generation_tokens_total{M}"] == 5
    assert s[f"tensorfold:prompt_tokens_total{M}"] == prompt
    assert s['tensorfold:prompt_tokens_by_source_total{model_name="fake-cuda",source="local_cache_hit"}'] == 3
    assert s['tensorfold:prompt_tokens_by_source_total{model_name="fake-cuda",source="local_compute"}'] == prompt - 3
    assert s[f"tensorfold:prompt_tokens_cached_total{M}"] == 3
    assert s[f"tensorfold:prefix_cache_queries_total{M}"] == prompt
    assert s[f"tensorfold:prefix_cache_hits_total{M}"] == 3
    assert s[f"tensorfold:time_to_first_token_seconds_count{M}"] == 1
    assert s[f"tensorfold:inter_token_latency_seconds_count{M}"] == 4        # one per round after the first
    assert s[f"tensorfold:request_time_per_output_token_seconds_count{M}"] == 1
    assert s[f"tensorfold:request_prompt_tokens_sum{M}"] == prompt
    assert s[f"tensorfold:request_generation_tokens_sum{M}"] == 5
    assert s[f"tensorfold:request_params_max_tokens_sum{M}"] == 5
    assert s[f"tensorfold:request_prefill_kv_computed_tokens_sum{M}"] == prompt - 3
    for name in ("e2e_request_latency", "request_queue_time", "request_prefill_time", "request_decode_time",
                 "request_inference_time"):
        assert s[f"tensorfold:{name}_seconds_count{M}"] == 1
    # the parts add up: queue + prefill + decode = e2e
    parts = sum(s[f"tensorfold:request_{p}_time_seconds_sum{M}"] for p in ("queue", "prefill", "decode"))
    assert parts == pytest.approx(s[f"tensorfold:e2e_request_latency_seconds_sum{M}"], abs=1e-6)
    assert s[f"tensorfold:num_requests_running{M}"] == 0
    assert s[f"tensorfold:num_requests_waiting{M}"] == 0


def test_an_end_token_finishes_as_stop(tmp_path):
    app = measured(tmp_path, PacedEngine("ab\x00"))                  # the reply ends on the end token
    assert app.run({"messages": MESSAGES, "max_tokens": 3}, True, lambda delta: True)["finish"] == "stop"
    assert success(scrape(app), "stop") == 1


def test_a_tool_call_finishes_as_stop_as_in_vllm(tmp_path, monkeypatch):
    from tensorfold.cuda import server

    app = measured(tmp_path, PacedEngine("ab\x00"))
    monkeypatch.setattr(server, "parse_tool_calls", lambda text, tools, **kw: ("", [{"id": "call_0"}]))
    body = {"messages": MESSAGES, "max_tokens": 3,
            "tools": [{"type": "function", "function": {"name": "measure", "parameters": {"type": "object"}}}]}
    assert app.run(body, True, lambda delta: True)["finish"] == "tool_calls"
    s = scrape(app)
    assert success(s, "stop") == 1 and sum(success(s, r) for r in metrics.FINISHED_REASONS) == 1


def test_a_client_that_leaves_is_an_abort_and_its_prompt_still_counts(tmp_path):
    app = measured(tmp_path, CachedEngine())
    with pytest.raises(RequestCancelled):
        app.run({"messages": MESSAGES, "max_tokens": 50}, True, lambda delta: False)
    s = scrape(app)
    assert success(s, "abort") == 1
    assert s[f"tensorfold:generation_tokens_total{M}"] == 1
    assert s[f"tensorfold:prompt_tokens_total{M}"] > 0          # the engine prefilled it
    assert s[f"tensorfold:num_requests_running{M}"] == 0


def test_an_engine_failure_is_an_error_with_unknown_prompt_counts(tmp_path):
    app = measured(tmp_path, FailingEngine())
    with pytest.raises(RuntimeError):
        app.run({"messages": MESSAGES, "max_tokens": 50}, True, lambda delta: True)
    s = scrape(app)
    assert success(s, "error") == 1
    assert s[f"tensorfold:prompt_tokens_total{M}"] == 0          # never guessed
    assert s[f"tensorfold:e2e_request_latency_seconds_count{M}"] == 0
    assert s[f"tensorfold:num_requests_running{M}"] == 0


def test_rounds_after_a_stop_are_not_counted(tmp_path):
    # two-rank GLM keeps decoding after on_tokens returns True; those rounds never reach the request
    app = measured(tmp_path, PacedEngine("abcd", finish_all=True))
    result = app.run({"messages": MESSAGES, "max_tokens": 30, "stop": ["bc"]}, True, lambda delta: True)
    assert result["finish"] == "stop"
    assert scrape(app)[f"tensorfold:generation_tokens_total{M}"] == 3


def test_a_request_waiting_for_the_serialized_engine_is_waiting(tmp_path):
    engine = PacedEngine(hold_at=2)
    app = measured(tmp_path, engine)
    body = {"messages": MESSAGES, "max_tokens": 4}
    runs = [threading.Thread(target=app.run, args=(body, True, lambda delta: True)) for _ in range(2)]
    runs[0].start()
    assert engine.held.wait(WAIT)
    runs[1].start()
    until(lambda: scrape(app)[f"tensorfold:num_requests_waiting{M}"] == 1, "the second request to wait")
    assert scrape(app)[f"tensorfold:num_requests_running{M}"] == 1
    engine.release.set()
    for run in runs:
        run.join(WAIT)
    s = scrape(app)
    assert s[f"tensorfold:num_requests_waiting{M}"] == 0 and s[f"tensorfold:num_requests_running{M}"] == 0
    assert success(s, "length") == 2
    assert s[f"tensorfold:request_queue_time_seconds_count{M}"] == 2


def test_a_parallel_engine_reports_its_own_queue_time(tmp_path):
    app = measured(tmp_path, SchedulerEngine())
    result = app.run({"messages": MESSAGES, "max_tokens": 4}, True, lambda delta: True)
    assert "queue_s" in result["stats"]
    s = scrape(app)
    assert success(s, "length") == 1
    assert s[f"tensorfold:request_queue_time_seconds_count{M}"] == 1
    assert s[f"tensorfold:generation_tokens_total{M}"] == 4


def test_the_parallel_engine_queue_is_waiting(tmp_path):
    engine = PacedEngine()
    engine.scheduler = SimpleNamespace(waiting=queue.Queue())   # a --parallel engine's queue, never admitted
    app = measured(tmp_path, engine)
    engine.scheduler.waiting.put(("stream", "box"))
    engine.scheduler.waiting.put(("stream", "box"))
    with app.metrics.lock:
        app.metrics.running = 3                              # three in flight, two of them still queued
    s = scrape(app)
    assert s[f"tensorfold:num_requests_waiting{M}"] == 2
    assert s[f"tensorfold:num_requests_running{M}"] == 1


def test_the_server_serves_metrics_as_prometheus_text(tmp_path):
    app = measured(tmp_path, PacedEngine())
    app.run({"messages": MESSAGES, "max_tokens": 3}, True, lambda delta: True)
    with serving(app) as port:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=WAIT)
        conn.request("GET", "/metrics")
        response = conn.getresponse()
        text = response.read().decode()
        conn.close()
    assert response.status == 200
    assert response.getheader("Content-Type") == "text/plain; version=0.0.4; charset=utf-8"
    assert samples(text)[f"tensorfold:generation_tokens_total{M}"] == 3


def test_an_app_without_metrics_serves_an_empty_exposition(tmp_path):
    app = app_for(tmp_path, PacedEngine())                   # built without __init__, as older tests do
    app.run({"messages": MESSAGES, "max_tokens": 3}, True, lambda delta: True)
    with serving(app) as port:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=WAIT)
        conn.request("GET", "/metrics")
        response = conn.getresponse()
        assert response.status == 200 and response.read() == b""
        conn.close()
