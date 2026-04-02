"""
Neural Granger Causality baseline (Tank et al., 2021).

Implements cLSTM: one LSTM per target variable with group-lasso
regularization on input weights to zero out non-causal connections.

This is a simplified but faithful implementation of the key idea.
For exact reproduction of Tank et al. results, use their public code.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class cLSTMCell(nn.Module):
    """
    Causal LSTM cell for one target variable.

    Input weights are split per source variable so that
    group lasso can zero out entire input channels.
    """

    def __init__(self, n_variables: int, hidden_size: int = 32):
        super().__init__()
        self.n_variables = n_variables
        self.hidden_size = hidden_size

        # Separate input weight per source variable (for group lasso)
        self.W_input = nn.ParameterList([
            nn.Parameter(torch.randn(4 * hidden_size, 1) * 0.01)
            for _ in range(n_variables)
        ])

        # Recurrent weights
        self.W_hidden = nn.Parameter(
            torch.randn(4 * hidden_size, hidden_size) * 0.01
        )
        self.bias = nn.Parameter(torch.zeros(4 * hidden_size))

    def forward(
        self, x: torch.Tensor, hc: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Input [batch, n_variables].
            hc: (h, c) tuple, each [batch, hidden_size].

        Returns:
            h_new, (h_new, c_new).
        """
        batch = x.shape[0]

        if hc is None:
            h = torch.zeros(batch, self.hidden_size, device=x.device)
            c = torch.zeros(batch, self.hidden_size, device=x.device)
        else:
            h, c = hc

        # Compute input contribution from each source variable separately
        gates = self.bias.clone().unsqueeze(0).expand(batch, -1)

        for j in range(self.n_variables):
            gates = gates + x[:, j : j + 1] @ self.W_input[j].T

        gates = gates + h @ self.W_hidden.T

        # Split into 4 gates
        i_gate, f_gate, g_gate, o_gate = gates.chunk(4, dim=-1)
        i_gate = torch.sigmoid(i_gate)
        f_gate = torch.sigmoid(f_gate)
        g_gate = torch.tanh(g_gate)
        o_gate = torch.sigmoid(o_gate)

        c_new = f_gate * c + i_gate * g_gate
        h_new = o_gate * torch.tanh(c_new)

        return h_new, (h_new, c_new)

    def get_input_weight_norms(self) -> torch.Tensor:
        """
        Get L2 norm of input weights for each source variable.
        Used for causal scoring: high norm = strong input influence.

        Returns:
            norms: [n_variables] tensor.
        """
        norms = torch.stack([w.norm() for w in self.W_input])
        return norms


class cLSTMGrangerCausality(nn.Module):
    """
    Full cLSTM model for Granger causality discovery.

    Trains K separate cLSTM cells, one per target variable.
    Uses group lasso on input weights to encourage sparsity.
    """

    def __init__(self, n_variables: int, hidden_size: int = 32):
        super().__init__()
        self.n_variables = n_variables
        self.hidden_size = hidden_size

        # One cLSTM cell per target variable
        self.cells = nn.ModuleList([
            cLSTMCell(n_variables, hidden_size) for _ in range(n_variables)
        ])

        # Output projection per target
        self.output_projs = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(n_variables)
        ])

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: Input [batch, seq_len, n_variables].

        Returns:
            pred: Predicted next step [batch, seq_len, n_variables].
        """
        batch, seq_len, K = X.shape
        outputs = []

        for i in range(K):
            h_c = None
            target_preds = []

            for t in range(seq_len):
                h, h_c = self.cells[i](X[:, t, :], h_c)
                pred_t = self.output_projs[i](h)
                target_preds.append(pred_t)

            outputs.append(torch.stack(target_preds, dim=1))  # [B, L, 1]

        return torch.cat(outputs, dim=-1)  # [B, L, K]

    def get_causal_score_matrix(self) -> np.ndarray:
        """
        Extract causal score matrix from input weight norms.

        S[i, j] = L2 norm of the input weights from variable j
                   in the cLSTM cell predicting variable i.

        Returns:
            S: Causal score matrix (K, K).
        """
        K = self.n_variables
        S = np.zeros((K, K))

        for i in range(K):
            norms = self.cells[i].get_input_weight_norms().detach().cpu().numpy()
            S[i, :] = norms

        # Normalize
        if S.max() > 0:
            S = S / S.max()

        return S

    def group_lasso_penalty(self, lambda_group: float = 0.01) -> torch.Tensor:
        """
        Compute group lasso penalty on input weights.

        This encourages entire input channels to be zeroed out,
        corresponding to absence of causal relationships.

        Args:
            lambda_group: Regularization strength.

        Returns:
            Penalty term (scalar tensor).
        """
        penalty = torch.tensor(0.0, device=next(self.parameters()).device)
        for cell in self.cells:
            for w in cell.W_input:
                penalty = penalty + w.norm()
        return lambda_group * penalty


def train_clstm(
    X: np.ndarray,
    hidden_size: int = 32,
    n_epochs: int = 200,
    lr: float = 1e-3,
    lambda_group: float = 0.01,
    seq_len: int = 200,
    batch_size: int = 16,
    device: str = "cpu",
    verbose: bool = True,
) -> Tuple[cLSTMGrangerCausality, np.ndarray]:
    """
    Train a cLSTM model and return the causal score matrix.

    This is the main entry point for the Neural Granger baseline.

    Args:
        X: Time series (T, K).
        hidden_size: LSTM hidden dimension.
        n_epochs: Training epochs.
        lr: Learning rate.
        lambda_group: Group lasso strength.
        seq_len: Training subsequence length.
        batch_size: Batch size.
        device: Compute device.
        verbose: Print progress.

    Returns:
        model: Trained cLSTM model.
        S: Causal score matrix (K, K).
    """
    T, K = X.shape
    X_tensor = torch.FloatTensor(X).to(device)

    model = cLSTMGrangerCausality(K, hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(n_epochs):
        # Sample random subsequences
        starts = torch.randint(0, T - seq_len - 1, (batch_size,))
        batch_x = torch.stack([X_tensor[s : s + seq_len] for s in starts])
        batch_y = torch.stack([X_tensor[s + 1 : s + seq_len + 1] for s in starts])

        pred = model(batch_x)
        mse_loss = loss_fn(pred, batch_y)
        reg_loss = model.group_lasso_penalty(lambda_group)
        loss = mse_loss + reg_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if verbose and (epoch + 1) % 50 == 0:
            print(
                f"cLSTM Epoch {epoch + 1:4d}/{n_epochs} | "
                f"MSE: {mse_loss.item():.6f} | Reg: {reg_loss.item():.6f}"
            )

    model.eval()
    S = model.get_causal_score_matrix()
    return model, S
