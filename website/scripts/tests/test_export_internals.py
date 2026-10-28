"""Checks on the /lab exports against the checkpoints and runs they were read from.

Run from the repository root with the root venv, which has torch and pennylane:

    .venv/Scripts/python.exe -m pytest website/scripts/tests/test_export_internals.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pennylane")

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import export_internals as ei  # noqa: E402

LAB = ei.OUT


def load(name: str) -> dict:
    return json.loads((LAB / name).read_text(encoding="utf-8"))


CHECKPOINTS = load("checkpoints.json")["checkpoints"]


@pytest.mark.parametrize("ck", CHECKPOINTS, ids=lambda c: c["path"].rsplit("/", 2)[-2] + "/" + c["path"].rsplit("/", 1)[-1])
def test_checkpoints_unchanged_since_export(ck):
    p = ei.REPO_ROOT / ck["path"]
    mtime = p.stat().st_mtime_ns
    assert ei.sha256(p) == ck["sha256"]
    assert p.stat().st_mtime_ns == mtime


def test_param_counts_match_the_runs():
    params = load("params.json")
    mars_cnn = json.loads((ei.MARS_CNN_DIR / "param_counts.json").read_text(encoding="utf-8"))
    mars_hyb = json.loads((ei.MARS_HYB_DIR / "param_counts.json").read_text(encoding="utf-8"))
    assert params["mars"]["cnn"]["trainable"] == mars_cnn["total_trainable"] == 11_242_563
    assert params["mars"]["hybrid"]["trainable"] == mars_hyb["total_trainable"] == 4_152
    assert params["mars"]["hybrid"]["bottleneck"] + params["mars"]["hybrid"]["qcnn_head"] == 4_152
    # The site's Moon summary (export_data.py, checked against the run by test_export_data.py).
    moon = json.loads((LAB.parent / "moon" / "summary.json").read_text(encoding="utf-8"))["parameter_counts"]
    assert params["moon"]["cnn"]["trainable_during_training"] == moon["cnn"]["trainable_during_training"] == 59_347
    assert params["moon"]["cnn"]["trainable_post_freeze"] == moon["cnn"]["trainable_recorded_post_freeze"] == 195
    assert params["moon"]["hybrid_head"] == moon["hybrid_head"] == 568
    assert params["moon"]["mlp_head"] == moon["mlp_head"] == 547


@pytest.mark.parametrize("world", ["mars", "moon"])
def test_rotations_match_checkpoint_tensors(world):
    rot = load(f"{world}_rotations.json")
    if world == "mars":
        state = torch.load(ei.MARS_HYB_DIR / "best_checkpoint.pt", map_location="cpu", weights_only=False)["model_state_dict"]
        prefix = "head."
    else:
        state = torch.load(ei.MOON_RUN / "checkpoints" / f"hybrid_seed_{ei.MOON_SEED}.pt", map_location="cpu", weights_only=False)["state_dict"]
        prefix = "qcnn."
    conv = state[prefix + "conv_weights"].numpy()
    pool = state[prefix + "pool_weights"].numpy()
    exported = [v for layer in rot["layers"] for v in layer["values"][0]]
    expected = list(conv.ravel()) + list(pool.ravel())
    assert exported == pytest.approx(expected, abs=1e-5)
    assert rot["readout"]["scale"] == pytest.approx(state[prefix + "out_scale"].numpy().tolist(), abs=1e-5)
    assert rot["readout"]["bias"] == pytest.approx(state[prefix + "out_bias"].numpy().tolist(), abs=1e-5)


def test_curves_cover_every_run_without_timing_fields():
    curves = load("curves.json")
    assert sorted(curves["mars"]) == sorted(f"{f}_{s}" for f in range(5) for s in (42, 43, 44))
    assert sorted(curves["moon"], key=int) == [str(s) for s in range(42, 52)]
    for run in curves["mars"].values():
        assert set(run) == {"cnn", "qcnn"}
        for c in run.values():
            assert set(c) == {"epoch", "train_loss", "val_loss"}
            assert c["epoch"][0] == 1 and len(c["epoch"]) == len(c["train_loss"]) == len(c["val_loss"])
    for run in curves["moon"].values():
        assert set(run) == {"cnn", "hybrid_qcnn", "frozen_embedding_mlp"}
        for c in run.values():
            assert set(c) == {"epoch", "train_loss", "val_macro_f1"}
            assert c["epoch"][0] == 1


def test_excerpts_equal_the_committed_source():
    ex = load("excerpts.json")
    commit = ex["commit"]
    for e in ex["excerpts"]:
        source = subprocess.run(["git", "show", f"{commit}:{e['path']}"], cwd=ei.REPO_ROOT, check=True,
                                capture_output=True, text=True, encoding="utf-8").stdout
        assert "\n".join(source.splitlines()[e["start"] - 1 : e["end"]]) == e["code"], e["id"]
        assert e["permalink"] == f"https://github.com/{ei.REPO}/blob/{commit}/{e['path']}#L{e['start']}-L{e['end']}"


def test_circuit_svg_is_themeable_text():
    svg = (LAB / "qcnn_circuit.svg").read_text(encoding="utf-8")
    assert svg.startswith("<svg") and "<metadata" not in svg
    assert "currentColor" in svg and "var(--circuit-fill)" in svg
    assert "#ffffff" not in svg and "#151515" not in svg
    assert ">RY<" in svg  # labels kept as text (svg.fonttype none)


@pytest.mark.skipif(not os.environ.get("LAB_FULL"), reason="re-runs the full export; set LAB_FULL=1")
def test_export_is_reproducible():
    before = {p.name: p.read_bytes() for p in LAB.iterdir()}
    ei.main()
    assert {p.name: p.read_bytes() for p in LAB.iterdir()} == before
