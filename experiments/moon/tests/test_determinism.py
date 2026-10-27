"""Determinism mechanics of the Moon trainer (CPU, no data needed).

Run from the repository root:

    .venv/Scripts/python.exe -m pytest experiments/moon/tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import train  # noqa: E402

CHECKPOINTS = HERE.parents[1] / "data" / "raw" / "moon" / "lrocnet" / "experiment_lunar_cnn_qcnn" / "runs" / "checkpoints"


def state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in module.state_dict().items()}


def same_state(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_stream_seeds_are_stable_and_distinct():
    seeds = {(s, n): train.stream_seed(s, n) for s in (42, 43) for n in train.STREAMS}
    assert len(set(seeds.values())) == len(seeds)
    assert train.stream_seed(42, "hybrid_init") == train.stream_seed(42, "hybrid_init")
    with pytest.raises(KeyError):
        train.stream_seed(42, "not_a_stream")


@pytest.mark.parametrize("factory,name", [(train.HybridHead, "hybrid_init"), (train.ClassicalHead, "mlp_init"),
                                          (train.CNN, "cnn_init")])
def test_initialisation_ignores_earlier_random_draws(factory, name):
    torch.manual_seed(0)
    with train.init_stream(42, name):
        first = state(factory())
    # Consume global randomness the way the old trainer did (loader iterations, extra draws).
    torch.manual_seed(123)
    torch.rand(1000)
    for _ in train.loader(TensorDataset(torch.zeros(10, 1)), 4, torch.Generator().manual_seed(5)):
        pass
    with train.init_stream(42, name):
        second = state(factory())
    assert same_state(first, second)
    with train.init_stream(43, name):
        other_seed = state(factory())
    assert not same_state(first, other_seed)


def test_init_stream_restores_the_global_generator():
    torch.manual_seed(7)
    expected = torch.rand(3)
    torch.manual_seed(7)
    with train.init_stream(42, "hybrid_init"):
        train.HybridHead()
    assert torch.equal(torch.rand(3), expected)


def test_loader_order_depends_only_on_its_stream():
    ds = TensorDataset(torch.arange(64).float())

    def order(seed):
        return torch.cat([x for (x,) in train.loader(ds, 8, train.stream_generator(seed, "head_train"), True)])

    torch.manual_seed(0)
    a = order(42)
    torch.manual_seed(999)
    torch.rand(5000)
    b = order(42)
    assert torch.equal(a, b)
    assert not torch.equal(a, order(43))


def test_global_mean_pool_matches_adaptive_avg_pool():
    x = torch.randn(4, 96, 7, 9, dtype=torch.float64)
    assert torch.allclose(train.GlobalMeanPool()(x), nn.AdaptiveAvgPool2d(1)(x), rtol=0, atol=1e-12)


@pytest.mark.skipif(not (CHECKPOINTS / "cnn_seed_42.pt").exists(), reason="reported-run checkpoints not present")
def test_reported_checkpoints_still_load():
    saved = torch.load(CHECKPOINTS / "cnn_seed_42.pt", map_location="cpu", weights_only=False)["state_dict"]
    model = train.CNN()
    assert model.load_state_dict(saved) is not None  # strict: same keys as with AdaptiveAvgPool2d
    assert sum(p.numel() for p in model.parameters()) == 59347
