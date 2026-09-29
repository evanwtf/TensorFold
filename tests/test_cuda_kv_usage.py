"""Each CUDA engine's ``kv_usage``: the cache positions its live requests hold, of the positions it holds."""

from types import SimpleNamespace

import pytest

pytest.importorskip("tokenizers")

from tests.test_cuda_geometry import allocations  # noqa: F401  (fixture: fake triton, so the modules import)


def stream(pos):
    return SimpleNamespace(st=SimpleNamespace(pos=pos))


def engine(module: str, cls: str, **fields):
    import importlib

    e = getattr(importlib.import_module(module), cls).__new__(getattr(importlib.import_module(module), cls))
    e.__dict__.update(fields)
    return e


@pytest.mark.torch
def test_qwen27_on_one_stream_reports_nothing(allocations):  # noqa: F811
    e = engine("tensorfold.families.qwen3_5.cuda.engine", "Qwen27Engine", multi=None, scheduler=None)
    assert e.kv_usage() is None                               # its state is local to generate


@pytest.mark.torch
def test_qwen27_parallel_sums_decoding_and_prefilling_streams(allocations):  # noqa: F811
    multi = SimpleNamespace(streams={1: stream(100), 2: stream(50)}, filling=[stream(7), SimpleNamespace(st=None)],
                            context=4096)
    e = engine("tensorfold.families.qwen3_5.cuda.engine", "Qwen27Engine", multi=multi,
               scheduler=SimpleNamespace(max_streams=4))
    assert e.kv_usage() == (157, 4 * 4096)


@pytest.mark.torch
def test_flash_next_one_stream_and_parallel(allocations):  # noqa: F811
    mod, cls = "tensorfold.families.qwen4_exp.cuda.engine", "FlashNextEngine"
    one = engine(mod, cls, multi=None, e=SimpleNamespace(st=SimpleNamespace(pos=300)), max_len=8192)
    assert one.kv_usage() == (300, 8192)
    par = engine(mod, cls, multi=SimpleNamespace(streams={1: stream(10), 2: stream(20)}),
                 scheduler=SimpleNamespace(max_streams=3), max_len=1000)
    assert par.kv_usage() == (30, 3000)


@pytest.mark.torch
def test_glm_and_nemotron(allocations):  # noqa: F811
    glm = engine("tensorfold.families.glm5_next.cuda.engine", "GlmEngine",
                 e=SimpleNamespace(st=SimpleNamespace(pos=12)), capacity_plan={"cache_slots": 65536})
    assert glm.kv_usage() == (12, 65536)
    nem = engine("tensorfold.families.nemotron_h.cuda.app", "NemotronEngine", e=SimpleNamespace(pos=9), max_len=512)
    assert nem.kv_usage() == (9, 512)
