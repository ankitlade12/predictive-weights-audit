"""
Lightweight Mamba-style SSM model for multivariate time series prediction.

This is a self-contained SSM implementation that does NOT require the
mamba_ssm CUDA kernels, making it runnable on CPU and any GPU.

For the research paper, we use this as the default model. If mamba_ssm
is installed, we also support wrapping the official Mamba-2 model.

The key design choice: B and C are data-dependent (selective), matching
the Mamba architecture's core innovation. This means B and C vary per
timestep, which is what makes causal score extraction non-trivial.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class SelectiveSSMLayer(nn.Module):
    """
    A single selective SSM layer (Mamba-style).

    Implements the core SSM recurrence:
        h[t] = A * h[t-1] + B[t] * x[t]
        y[t] = C[t] * h[t]

    where B[t] and C[t] are data-dependent (selective).

    Key attributes (accessible for causal extraction):
        self._B: Last computed B tensor [batch, seq, state_dim]
        self._C: Last computed C tensor [batch, seq, state_dim]
        self._gate: Last computed gate values [batch, seq, d_model]
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand

        if dt_rank == "auto":
            self.dt_rank = max(1, d_model // 16)
        else:
            self.dt_rank = int(dt_rank)

        # Input projection: x -> (z, x_ssm) split
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # 1D convolution (causal)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner,
        )

        # SSM parameter projections (data-dependent B, C, dt)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # A parameter: learnable, initialized as negative log-spaced
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))

        # D parameter (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # Caches for extraction
        self._B = None
        self._C = None
        self._gate = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, seq_len, d_model].

        Returns:
            Output tensor [batch, seq_len, d_model].
        """
        batch, seq_len, _ = x.shape

        # Input projection and split into SSM input and gate
        xz = self.in_proj(x)  # [B, L, 2 * d_inner]
        x_ssm, z = xz.chunk(2, dim=-1)  # each [B, L, d_inner]

        # 1D causal convolution
        x_conv = x_ssm.transpose(1, 2)  # [B, d_inner, L]
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]  # causal: trim right
        x_conv = x_conv.transpose(1, 2)  # [B, L, d_inner]
        x_conv = F.silu(x_conv)

        # Compute data-dependent SSM parameters
        x_proj = self.x_proj(x_conv)  # [B, L, dt_rank + 2*d_state]
        dt, B, C = x_proj.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        # Store B, C for causal extraction
        self._B = B.detach()  # [B, L, d_state]
        self._C = C.detach()  # [B, L, d_state]

        # Discretize: dt -> delta
        dt = self.dt_proj(dt)  # [B, L, d_inner]
        dt = F.softplus(dt)

        # A matrix (negative exponential for stability)
        A = -torch.exp(self.A_log)  # [d_inner, d_state]

        # SSM recurrence (scan)
        y = self._selective_scan(x_conv, dt, A, B, C)  # [B, L, d_inner]

        # Skip connection
        y = y + x_conv * self.D.unsqueeze(0).unsqueeze(0)

        # Gate and output
        gate = F.silu(z)
        self._gate = gate.detach()
        y = y * gate

        return self.out_proj(y)

    def _selective_scan(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the selective SSM scan (sequential for correctness).

        For production, this would use the parallel scan from mamba_ssm.
        For research/small-scale, sequential scan is fine.

        Args:
            x: [batch, seq_len, d_inner]
            dt: [batch, seq_len, d_inner] — discretization step
            A: [d_inner, d_state] — state transition (continuous)
            B: [batch, seq_len, d_state] — input projection
            C: [batch, seq_len, d_state] — output projection

        Returns:
            y: [batch, seq_len, d_inner]
        """
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[1]

        # Discretize A and B
        # dA = exp(dt * A), dB = dt * B
        # dt: [B, L, d_inner], A: [d_inner, d_state]
        dA = torch.exp(
            dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
        )  # [B, L, d_inner, d_state]

        # dB * x: [B, L, d_inner, d_state]
        dBx = (
            dt.unsqueeze(-1)
            * B.unsqueeze(2)  # [B, L, 1, d_state]
            * x.unsqueeze(-1)  # [B, L, d_inner, 1]
        )

        # Sequential scan
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        ys = []

        for t in range(seq_len):
            h = dA[:, t] * h + dBx[:, t]  # [B, d_inner, d_state]
            y_t = (h * C[:, t].unsqueeze(1)).sum(-1)  # [B, d_inner]
            ys.append(y_t)

        return torch.stack(ys, dim=1)  # [B, L, d_inner]


class MambaForTimeSeries(nn.Module):
    """
    Mamba model for multivariate time series next-step prediction.

    Architecture:
        Input projection (K -> d_model)
        -> N x SelectiveSSMLayer
        -> Output projection (d_model -> K)

    Training objective: MSE on next-step prediction.
    """

    def __init__(
        self,
        n_variables: int,
        d_model: int = 64,
        d_state: int = 16,
        n_layers: int = 4,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_variables = n_variables
        self.d_model = d_model
        self.n_layers = n_layers

        # Input: project K variables to d_model
        self.input_proj = nn.Linear(n_variables, d_model)

        # SSM layers
        self.layers = nn.ModuleList([
            SelectiveSSMLayer(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            for _ in range(n_layers)
        ])

        # Layer norms (pre-norm architecture)
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        # Output: project d_model back to K variables
        self.output_proj = nn.Linear(d_model, n_variables)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights following Mamba conventions."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input time series [batch, seq_len, n_variables].

        Returns:
            Predicted next step [batch, seq_len, n_variables].
        """
        h = self.input_proj(x)  # [B, L, d_model]

        for norm, layer in zip(self.norms, self.layers):
            residual = h
            h = norm(h)
            h = layer(h)
            h = self.dropout(h)
            h = h + residual

        return self.output_proj(h)  # [B, L, K]

    def get_ssm_layers(self):
        """Return list of SelectiveSSMLayer modules (for hook registration)."""
        return list(self.layers)


class SSMTrainer:
    """
    Training loop for the Mamba time series model.

    Trains on next-step prediction: given X[0:T-1], predict X[1:T].
    """

    def __init__(
        self,
        model: MambaForTimeSeries,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=lr * 0.01
        )
        self.loss_fn = nn.MSELoss()
        self.train_losses = []

    def train(
        self,
        X: torch.Tensor,
        n_epochs: int = 100,
        batch_size: int = 32,
        seq_len: int = 200,
        verbose: bool = True,
        print_every: int = 10,
    ) -> list:
        """
        Train the model on next-step prediction.

        Args:
            X: Full time series [T, K] as tensor.
            n_epochs: Number of training epochs.
            batch_size: Number of subsequences per batch.
            seq_len: Length of each training subsequence.
            verbose: Print training progress.
            print_every: Print every N epochs.

        Returns:
            List of per-epoch losses.
        """
        if isinstance(X, torch.Tensor):
            X = X.float().to(self.device)
        else:
            X = torch.FloatTensor(X).to(self.device)

        T, K = X.shape
        self.model.train()

        for epoch in range(n_epochs):
            epoch_loss = 0.0
            n_batches = 0

            # Sample random subsequences
            for _ in range(max(1, T // (batch_size * seq_len))):
                # Random start indices
                starts = torch.randint(0, T - seq_len - 1, (batch_size,))
                batch_x = torch.stack([X[s : s + seq_len] for s in starts])
                batch_y = torch.stack([X[s + 1 : s + seq_len + 1] for s in starts])

                # Forward pass
                pred = self.model(batch_x)

                loss = self.loss_fn(pred, batch_y)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            self.train_losses.append(avg_loss)
            self.scheduler.step()

            if verbose and (epoch + 1) % print_every == 0:
                print(f"Epoch {epoch + 1:4d}/{n_epochs} | Loss: {avg_loss:.6f}")

        return self.train_losses

    def plot_loss(self, save_path: Optional[str] = None):
        """Plot training loss curve."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(self.train_losses, linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_title("Training Loss")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300)
        return fig
