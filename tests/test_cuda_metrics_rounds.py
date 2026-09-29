"""A decoder's drafted rounds reach /metrics: the 27B's concurrent decoder, with its model replaced by a script."""

import importlib

import pytest

pytest.importorskip("tokenizers")

from tensorfold.cuda import metrics
from tensorfold.cuda.metrics import Metrics
from tensorfold.cuda.streams import Stream
from tests.test_cuda_27b_ignore_eos import SCRIPT, scripted_decoder
from tests.test_cuda_geometry import allocations  # noqa: F401  (fixture: fake triton, so the modules import)
from tests.test_cuda_metrics import samples

M = '{model_name="m"}'


@pytest.fixture
def registry():
    m = Metrics("m")
    metrics.install(m)
    yield m
    metrics.install(None)


def decode(draft: bool, count: int = 12) -> Stream:
    multi = importlib.import_module("tensorfold.families.qwen3_5.cuda.multi")
    dec = scripted_decoder(multi)
    s = Stream([3, 4], count, draft=draft, stop_eos=False)
    dec.admit(s)
    dec.finish([s] if s.done else [])
    while dec.live():
        dec.finish(dec.round())
    return s


@pytest.mark.torch
def test_drafted_rounds_count_drafts_and_accepted_tokens(allocations, registry):  # noqa: F811
    s = decode(draft=True)
    assert s.out == SCRIPT[:12]
    got = samples(registry.render())
    # the prefill gives the first token and every round one sampled token after its accepted drafts
    accepted = len(s.out) - 1 - s.rounds
    assert got[f"tensorfold:spec_decode_num_drafts_total{M}"] == s.rounds
    assert got[f"tensorfold:spec_decode_num_accepted_tokens_total{M}"] == accepted
    assert got[f"tensorfold:spec_decode_num_draft_tokens_total{M}"] >= accepted
    assert got['tensorfold:spec_decode_num_accepted_tokens_per_pos_total{model_name="m",position="0"}'] > 0


@pytest.mark.torch
def test_serial_rounds_draft_nothing(allocations, registry):  # noqa: F811
    decode(draft=False)
    assert "spec_decode" not in registry.render()


@pytest.mark.torch
def test_a_decoder_without_a_server_records_nothing(allocations):  # noqa: F811
    metrics.install(None)                                # rank 1, which never builds an App
    decode(draft=True)
    assert "spec_decode" not in Metrics("m").render()
