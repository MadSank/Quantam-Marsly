"""Tests for PennyLane QCNN batching equivalence and parameter matching against controls."""

import torch

from src.models.bottleneck_qcnn import BottleneckQCNNModel
from src.models.cnn import make_resnet_input_stem
from src.quantum.circuits import BatchedQCNNHead
from src.utils import count_parameters, load_config


def test_random_three_channel_resnet_stem_initializes_and_uses_every_channel() -> None:
    torch.manual_seed(137)
    original = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False)
    stem = make_resnet_input_stem(original, 3, pretrained=False)
    assert stem is original
    assert torch.all(stem.weight.flatten(2).norm(dim=2) > 0)

    for channel in range(3):
        image = torch.zeros(1, 3, 9, 9)
        image[:, channel] = 1.0
        contribution = stem(image)
        assert torch.isfinite(contribution).all()
        assert contribution.abs().sum().item() > 0


def test_resnet_stem_keeps_pretrained_three_channel_and_averages_only_pretrained_gray() -> None:
    old = torch.nn.Conv2d(3, 4, kernel_size=3, bias=False)
    with torch.no_grad():
        old.weight[:, 0].fill_(1.0)
        old.weight[:, 1].fill_(2.0)
        old.weight[:, 2].fill_(3.0)
    assert make_resnet_input_stem(old, 3, pretrained=True) is old
    gray = make_resnet_input_stem(old, 1, pretrained=True)
    assert gray.weight.shape[1] == 1
    assert torch.allclose(gray.weight, torch.full_like(gray.weight, 2.0))

    torch.manual_seed(139)
    random_gray = make_resnet_input_stem(old, 1, pretrained=False)
    assert random_gray is not old
    assert torch.all(random_gray.weight.flatten(2).norm(dim=2) > 0)


def test_qcnn_batching_matches_per_sample_loop() -> None:
    """Verifies PennyLane batched QNode forward pass matches sequential single-sample loop at atol=1e-5."""
    # 1. 8-qubit angle QCNN
    head_8 = BatchedQCNNHead(n_qubits=8, encoding="angle", use_entanglement=True)
    x_8 = torch.randn(6, 8, dtype=torch.float32)

    batch_out_8 = head_8(x_8)
    loop_out_8 = head_8.forward_per_sample_loop(x_8)

    assert torch.allclose(batch_out_8, loop_out_8, atol=1e-5), (
        f"8-qubit batched output differed from sequential loop: max diff = "
        f"{torch.max(torch.abs(batch_out_8 - loop_out_8)).item()}"
    )

    # 2. 10-qubit amplitude+sun QCNN
    head_10 = BatchedQCNNHead(n_qubits=10, encoding="amplitude", use_entanglement=True)
    x_10 = torch.randn(6, 256, dtype=torch.float32)
    x_10_norm = x_10 / torch.linalg.norm(x_10, dim=1, keepdim=True)
    sun_rad = torch.randn(6, 2, dtype=torch.float32)

    batch_out_10 = head_10(x_10_norm, sun_angles_rad=sun_rad)
    loop_out_10 = head_10.forward_per_sample_loop(x_10_norm, sun_angles_rad=sun_rad)

    assert torch.allclose(batch_out_10, loop_out_10, atol=1e-5), (
        f"10-qubit batched output differed from sequential loop: max diff = "
        f"{torch.max(torch.abs(batch_out_10 - loop_out_10)).item()}"
    )


def test_parameter_counts_logged_and_matched_mlp_within_10_percent() -> None:
    """Verifies Model 2 MLP control head and total parameter counts are within 10% of QCNN counts."""
    cfg = load_config("configs/default.yaml")

    qcnn_model = BottleneckQCNNModel(config=cfg, variant="qcnn", use_sun=True)
    mlp_model = BottleneckQCNNModel(config=cfg, variant="mlp_control", use_sun=True)

    qcnn_counts = count_parameters(qcnn_model)
    mlp_counts = count_parameters(mlp_model)

    n_qcnn_head = qcnn_counts["head_trainable"]
    n_mlp_head = mlp_counts["head_trainable"]

    n_qcnn_total = qcnn_counts["total_trainable"]
    n_mlp_total = mlp_counts["total_trainable"]

    # Head parameters: QCNN has 48, MLP has 51
    head_diff_ratio = abs(n_mlp_head - n_qcnn_head) / float(n_qcnn_head)
    assert head_diff_ratio <= 0.10, (
        f"Model 2 head parameter mismatch > 10%: QCNN={n_qcnn_head}, MLP={n_mlp_head} "
        f"(ratio={head_diff_ratio:.4f})"
    )

    # Total parameters: QCNN has 4176, MLP has 4179
    total_diff_ratio = abs(n_mlp_total - n_qcnn_total) / float(n_qcnn_total)
    assert total_diff_ratio <= 0.10, (
        f"Model 2 total parameter mismatch > 10%: QCNN={n_qcnn_total}, MLP={n_mlp_total} "
        f"(ratio={total_diff_ratio:.4f})"
    )
