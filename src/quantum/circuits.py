"""Batched PennyLane QCNN circuits and Quanvolution filter."""

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn


def apply_conv_layer(weights: torch.Tensor, wires: list[int], use_entanglement: bool = True) -> None:
    """Applies even-odd swept 2-qubit convolutional gates."""
    w_even = weights[0:6]
    for i in range(0, len(wires) - 1, 2):
        w0, w1 = wires[i], wires[i + 1]
        qml.RY(w_even[0], wires=w0)
        qml.RZ(w_even[1], wires=w0)
        qml.RY(w_even[2], wires=w1)
        qml.RZ(w_even[3], wires=w1)
        if use_entanglement:
            qml.CNOT(wires=[w0, w1])
        qml.RY(w_even[4], wires=w0)
        qml.RZ(w_even[5], wires=w1)
        if use_entanglement:
            qml.CNOT(wires=[w1, w0])

    w_odd = weights[6:12]
    if len(wires) > 2:
        for i in range(1, len(wires), 2):
            w0 = wires[i]
            w1 = wires[(i + 1) % len(wires)]
            qml.RY(w_odd[0], wires=w0)
            qml.RZ(w_odd[1], wires=w0)
            qml.RY(w_odd[2], wires=w1)
            qml.RZ(w_odd[3], wires=w1)
            if use_entanglement:
                qml.CNOT(wires=[w0, w1])
            qml.RY(w_odd[4], wires=w0)
            qml.RZ(w_odd[5], wires=w1)
            if use_entanglement:
                qml.CNOT(wires=[w1, w0])
    elif len(wires) == 2:
        w0, w1 = wires[0], wires[1]
        qml.RY(w_odd[0], wires=w0)
        qml.RZ(w_odd[1], wires=w0)
        qml.RY(w_odd[2], wires=w1)
        qml.RZ(w_odd[3], wires=w1)
        if use_entanglement:
            qml.CNOT(wires=[w0, w1])
        qml.RY(w_odd[4], wires=w0)
        qml.RZ(w_odd[5], wires=w1)
        if use_entanglement:
            qml.CNOT(wires=[w1, w0])


def apply_pool_layer(
    weights: torch.Tensor,
    control_wires: list[int],
    target_wires: list[int],
    use_entanglement: bool = True,
) -> None:
    """Applies pooling operations across control-target wire pairs."""
    for w_ctrl, w_tgt in zip(control_wires, target_wires):
        qml.RZ(weights[0], wires=w_ctrl)
        qml.RY(weights[1], wires=w_tgt)
        if use_entanglement:
            qml.CNOT(wires=[w_ctrl, w_tgt])
        qml.RY(weights[2], wires=w_tgt)


class BatchedQCNNHead(nn.Module):
    """Batched QCNN circuit module supporting angle and amplitude embeddings."""

    def __init__(
        self,
        n_qubits: int = 8,
        encoding: str = "angle",
        use_entanglement: bool = True,
        qubit_permutation: list[int] | None = None,
        n_conv_layers: int = 3,
    ) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.encoding = encoding
        self.use_entanglement = use_entanglement
        self.qubit_permutation = qubit_permutation
        self.n_conv_layers = n_conv_layers

        # Quantum parameters
        self.conv_weights = nn.Parameter(0.1 * torch.randn(n_conv_layers, 12))
        self.pool_weights = nn.Parameter(0.1 * torch.randn(2, 3))

        # Classical readout affine parameters
        self.out_scale = nn.Parameter(torch.ones(3))
        self.out_bias = nn.Parameter(torch.zeros(3))

        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def _circuit(
            features: torch.Tensor,
            conv_w: torch.Tensor,
            pool_w: torch.Tensor,
            sun_angles: torch.Tensor | None = None,
        ):
            # Input encoding
            if self.encoding == "angle":
                feat = features
                if self.qubit_permutation is not None:
                    feat = feat[..., self.qubit_permutation]
                for q in range(self.n_qubits):
                    qml.RY(feat[..., q], wires=q)
            elif self.encoding == "amplitude":
                qml.AmplitudeEmbedding(features=features, wires=range(8), normalize=False)
                if self.n_qubits == 10 and sun_angles is not None:
                    qml.RY(sun_angles[..., 0], wires=8)
                    qml.RY(sun_angles[..., 1], wires=9)
            else:
                raise ValueError(f"Unsupported encoding: {self.encoding}")

            # Wire reduction schedule
            if self.n_qubits == 8:
                # Stage 0: 8 qubits -> Conv 0 -> Pool 0 -> 4 qubits [1, 3, 5, 7]
                apply_conv_layer(conv_w[0], wires=list(range(8)), use_entanglement=self.use_entanglement)
                apply_pool_layer(
                    pool_w[0],
                    control_wires=[0, 2, 4, 6],
                    target_wires=[1, 3, 5, 7],
                    use_entanglement=self.use_entanglement,
                )

                # Stage 1: 4 qubits -> Conv 1 -> Pool 1 -> 2 qubits [3, 7]
                apply_conv_layer(conv_w[1], wires=[1, 3, 5, 7], use_entanglement=self.use_entanglement)
                apply_pool_layer(
                    pool_w[1],
                    control_wires=[1, 5],
                    target_wires=[3, 7],
                    use_entanglement=self.use_entanglement,
                )

                # Stage 2: 2 qubits -> Conv 2
                apply_conv_layer(conv_w[2], wires=[3, 7], use_entanglement=self.use_entanglement)

                return [
                    qml.expval(qml.PauliZ(3)),
                    qml.expval(qml.PauliZ(7)),
                    qml.expval(qml.PauliZ(3) @ qml.PauliZ(7)),
                ]

            elif self.n_qubits == 10:
                # Stage 0: 10 qubits -> Conv 0 -> Pool 0 -> 5 qubits [1, 3, 5, 7, 9]
                apply_conv_layer(conv_w[0], wires=list(range(10)), use_entanglement=self.use_entanglement)
                apply_pool_layer(
                    pool_w[0],
                    control_wires=[0, 2, 4, 6, 8],
                    target_wires=[1, 3, 5, 7, 9],
                    use_entanglement=self.use_entanglement,
                )

                # Stage 1: 5 qubits -> Conv 1 -> Pool 1 -> 2 qubits [3, 9]
                apply_conv_layer(conv_w[1], wires=[1, 3, 5, 7, 9], use_entanglement=self.use_entanglement)
                apply_pool_layer(
                    pool_w[1],
                    control_wires=[1, 5, 7],
                    target_wires=[3, 9, 9],
                    use_entanglement=self.use_entanglement,
                )

                # Stage 2: 2 qubits -> Conv 2
                apply_conv_layer(conv_w[2], wires=[3, 9], use_entanglement=self.use_entanglement)

                return [
                    qml.expval(qml.PauliZ(3)),
                    qml.expval(qml.PauliZ(9)),
                    qml.expval(qml.PauliZ(3) @ qml.PauliZ(9)),
                ]
            else:
                raise ValueError(f"Unsupported n_qubits: {self.n_qubits}")

        self.circuit = _circuit

    def forward(self, x: torch.Tensor, sun_angles_rad: torch.Tensor | None = None) -> torch.Tensor:
        """Executes batched QNode across B samples simultaneously without Python loops."""
        expvals = self.circuit(x, self.conv_weights, self.pool_weights, sun_angles_rad)
        stacked = torch.stack(expvals, dim=-1).to(dtype=torch.float32)
        return stacked * self.out_scale + self.out_bias

    def forward_per_sample_loop(
        self, x: torch.Tensor, sun_angles_rad: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Sequential single-sample reference implementation used only by pytest for equivalence testing."""
        batch_size = x.shape[0]
        outputs = []
        for i in range(batch_size):
            x_i = x[i : i + 1]
            s_i = sun_angles_rad[i : i + 1] if sun_angles_rad is not None else None
            out_i = self.forward(x_i, s_i)
            outputs.append(out_i)
        return torch.cat(outputs, dim=0)


class FixedQuanvolutionFilter(nn.Module):
    """Cached 4-qubit 2x2 quanvolutional filter with fixed random unitary layers."""

    def __init__(self, n_layers: int = 2, seed: int = 2026) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.seed = seed
        self.dev = qml.device("default.qubit", wires=4)

        rng = np.random.default_rng(seed)
        weights = torch.tensor(
            rng.uniform(0.0, 2.0 * np.pi, size=(n_layers, 4)), dtype=torch.float32
        )
        self.register_buffer("weights", weights)

        @qml.qnode(self.dev, interface="torch", diff_method=None)
        def _circuit(patch: torch.Tensor, weights: torch.Tensor):
            for j in range(4):
                qml.RY(torch.pi * patch[..., j], wires=j)
            qml.RandomLayers(weights, wires=range(4), seed=seed)
            return [qml.expval(qml.PauliZ(j)) for j in range(4)]

        self._circuit = _circuit

    def forward(self, patches: torch.Tensor, chunk_size: int = 4096) -> torch.Tensor:
        """Evaluates patches (N, 4) in batched chunks under torch.no_grad() -> (N, 4)."""
        patches = patches.to(dtype=torch.float32)
        n_total = patches.shape[0]
        outputs = []
        with torch.no_grad():
            for start_idx in range(0, n_total, chunk_size):
                chunk = patches[start_idx : start_idx + chunk_size]
                expvals = self._circuit(chunk, self.weights)
                stacked = torch.stack(expvals, dim=-1)
                outputs.append(stacked)
        return torch.cat(outputs, dim=0)
