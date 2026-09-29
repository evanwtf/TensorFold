"""The CUDA server's Prometheus metrics: vLLM's families, names, buckets and labels, under ``tensorfold:``."""

import pytest

from tensorfold.cuda import metrics
from tensorfold.cuda.metrics import Metrics


def samples(text: str) -> dict[str, float]:
    """``name{labels}`` to value, for every sample line of an exposition."""

    out = {}
    for line in text.splitlines():
        if line and not line.startswith("#"):
            key, value = line.rsplit(" ", 1)
            out[key] = float(value)
    return out


def types(text: str) -> dict[str, str]:
    return {line.split()[2]: line.split()[3] for line in text.splitlines() if line.startswith("# TYPE ")}


@pytest.fixture(autouse=True)
def no_sink():
    metrics.install(None)
    yield
    metrics.install(None)


def test_a_fresh_server_exports_zero_counters_under_vllm_names():
    text = Metrics("m").render()
    s = samples(text)
    assert s['tensorfold:prompt_tokens_total{model_name="m"}'] == 0
    assert s['tensorfold:generation_tokens_total{model_name="m"}'] == 0
    assert s['tensorfold:num_requests_running{model_name="m"}'] == 0
    assert s['tensorfold:num_requests_waiting{model_name="m"}'] == 0
    assert s['tensorfold:time_to_first_token_seconds_count{model_name="m"}'] == 0
    assert s['tensorfold:prompt_tokens_by_source_total{model_name="m",source="local_compute"}'] == 0
    assert s['tensorfold:prompt_tokens_by_source_total{model_name="m",source="local_cache_hit"}'] == 0
    for reason in ("stop", "length", "abort", "error"):
        assert s[f'tensorfold:request_success_total{{model_name="m",finished_reason="{reason}"}}'] == 0
    t = types(text)
    assert t["tensorfold:prompt_tokens"] == "counter"          # a counter's TYPE line drops _total, as prometheus_client
    assert t["tensorfold:num_requests_running"] == "gauge"
    assert t["tensorfold:e2e_request_latency_seconds"] == "histogram"


def test_series_an_engine_feeds_are_absent_until_it_feeds_them():
    text = Metrics("m").render()
    assert "spec_decode" not in text                          # unavailable is not zero
    assert "kv_cache_usage_perc" not in text


def test_histogram_buckets_are_vllm_bounds_cumulative_with_inf():
    m = Metrics("m")
    m.observe("time_to_first_token_seconds", 0.003)
    m.observe("time_to_first_token_seconds", 3000.0)
    s = samples(m.render())
    ttft = 'tensorfold:time_to_first_token_seconds_bucket{model_name="m",le="%s"}'
    assert s[ttft % "0.001"] == 0
    assert s[ttft % "0.005"] == 1
    assert s[ttft % "2560.0"] == 1                            # vLLM's largest TTFT bound
    assert s[ttft % "+Inf"] == 2
    assert s['tensorfold:time_to_first_token_seconds_sum{model_name="m"}'] == pytest.approx(3000.003)
    assert s['tensorfold:time_to_first_token_seconds_count{model_name="m"}'] == 2


def test_bucket_bounds_match_a_live_vllm_capture():
    # read from a vLLM server's /metrics (llm-metrics-exporter testdata/vllm), not from vLLM's source
    assert metrics.LATENCY[0] == 0.3 and metrics.LATENCY[-1] == 7680.0 and len(metrics.LATENCY) == 21
    assert metrics.TTFT[0] == 0.001 and metrics.TTFT[-1] == 2560.0 and len(metrics.TTFT) == 22
    assert metrics.ITL[0] == 0.01 and metrics.ITL[-1] == 80.0 and len(metrics.ITL) == 19
    assert metrics.TOKENS[:4] == (1, 2, 5, 10) and metrics.TOKENS[-1] == 200000 and len(metrics.TOKENS) == 17


def test_a_drafted_round_counts_drafts_tokens_and_accepted_positions():
    m = Metrics("m")
    metrics.install(m)
    metrics.round(drafted=4, accepted=2)
    metrics.round(drafted=4, accepted=0)
    metrics.round(drafted=0, accepted=0)                      # an undrafted round is no draft
    s = samples(m.render())
    assert s['tensorfold:spec_decode_num_drafts_total{model_name="m"}'] == 2
    assert s['tensorfold:spec_decode_num_draft_tokens_total{model_name="m"}'] == 8
    assert s['tensorfold:spec_decode_num_accepted_tokens_total{model_name="m"}'] == 2
    per_pos = 'tensorfold:spec_decode_num_accepted_tokens_per_pos_total{model_name="m",position="%d"}'
    assert s[per_pos % 0] == 1 and s[per_pos % 1] == 1    # vLLM numbers positions from 0
    assert s[per_pos % 2] == 0 and s[per_pos % 3] == 0        # proposed and never accepted: a real zero
    assert per_pos % 4 not in s


def test_a_tree_draft_numbers_positions_by_depth():
    m = Metrics("m")
    metrics.install(m)
    metrics.round(drafted=40, accepted=3, positions=6)        # 40 tree nodes, 6 deep
    s = samples(m.render())
    assert s['tensorfold:spec_decode_num_draft_tokens_total{model_name="m"}'] == 40
    per_pos = 'tensorfold:spec_decode_num_accepted_tokens_per_pos_total{model_name="m",position="%d"}'
    assert [s[per_pos % i] for i in range(6)] == [1, 1, 1, 0, 0, 0]
    assert per_pos % 6 not in s


def test_round_without_an_installed_registry_records_nothing():
    m = Metrics("m")
    metrics.round(drafted=4, accepted=4)                      # rank 1, the Mac, a unit test: no server
    assert "spec_decode" not in m.render()


def test_kv_usage_is_read_from_the_engine_at_scrape_time():
    class Engine:
        used = 250

        def kv_usage(self):
            return self.used, 1000

    engine = Engine()
    m = Metrics("m")
    assert samples(m.render(engine))['tensorfold:kv_cache_usage_perc{model_name="m"}'] == 0.25
    engine.used = 500
    assert samples(m.render(engine))['tensorfold:kv_cache_usage_perc{model_name="m"}'] == 0.5


def test_an_engine_without_kv_usage_or_capacity_exports_no_kv_series():
    class NoCapacity:
        def kv_usage(self):
            return 0, 0

    assert "kv_cache_usage_perc" not in Metrics("m").render(object())
    assert "kv_cache_usage_perc" not in Metrics("m").render(NoCapacity())


def test_label_values_are_escaped():
    text = Metrics('a"b\\c\nd').render()
    assert 'model_name="a\\"b\\\\c\\nd"' in text


def test_every_family_has_help_and_type_before_its_samples():
    m = Metrics("m")
    metrics.install(m)
    metrics.round(drafted=2, accepted=1)
    seen = set()
    for line in m.render().splitlines():
        if line.startswith("# TYPE "):
            seen.add(line.split()[2])
        elif not line.startswith("#"):
            name = line.split("{")[0]
            base = next(b for b in (name, name.removesuffix("_total"), name.rsplit("_", 1)[0]) if b in seen)
            assert base in seen
