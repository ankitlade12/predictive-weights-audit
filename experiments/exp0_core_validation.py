"""
Core Validation Experiment — Day 3 Test
========================================

THE most important experiment in this entire project.

Question: Does the trained Mamba model's internal structure encode
the true causal graph of a VAR(1) process?

Two extraction methods:
    1. Jacobian-based: dY_i/dX_j averaged over timesteps (gold standard, slower)
    2. Weight-projection: W_out @ W_in from the trained model (fast approximation)

If either recovers causal structure (AUROC > 0.7) → the paper is real.

Usage:
    python experiments/exp0_core_validation.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.synthetic import generate_var1
from ssm_causal.mamba_model import MambaForTimeSeries, SSMTrainer
from ssm_causal.utils import compute_metrics, plot_comparison, print_metrics_table


def extract_jacobian_score(model, X_tensor, device="cpu", seq_len=100):
    """
    Compute causal score via input-output Jacobian.

    For each output variable i, backprop through the model and measure
    how much each input variable j contributes (averaged over time).

    This is the gold-standard extraction — captures the full computational
    graph including B, C, projections, gates, everything.
    """
    model.eval()
    K = X_tensor.shape[1]

    X_sub = X_tensor[:seq_len].unsqueeze(0).to(device).requires_grad_(True)
    Y = model(X_sub)  # [1, seq_len, K]

    S = torch.zeros(K, K)
    for i in range(K):
        model.zero_grad()
        if X_sub.grad is not None:
            X_sub.grad.zero_()

        Y[0, :, i].sum().backward(retain_graph=(i < K - 1))
        S[i, :] = X_sub.grad[0].abs().mean(dim=0)

    S = S.detach().numpy()
    np.fill_diagonal(S, 0)
    if S.max() > 0:
        S = S / S.max()
    return S


def extract_weight_score(model):
    """
    Fast causal score from the model's learned projection weights.

    The effective variable-to-variable mapping is:
        S = |W_out @ W_in|
    where W_in: [d_model, K] and W_out: [K, d_model].

    This captures which input variables the model has learned to
    route to which output variables through the d_model bottleneck.
    """
    W_in = model.input_proj.weight.detach().cpu()   # [d_model, K]
    W_out = model.output_proj.weight.detach().cpu()  # [K, d_model]

    S = (W_out @ W_in).abs().numpy()  # [K, K]
    np.fill_diagonal(S, 0)
    if S.max() > 0:
        S = S / S.max()
    return S


def extract_bc_projected_score(model, X_tensor, device="cpu"):
    """
    B·C^T projected through input/output weights.

    Full causal path: X -> W_in -> [SSM with B,C] -> W_out -> Y
    We extract B, C from each layer's forward pass, compute the
    outer product, and project through the endpoint matrices.

    This is the theoretical core of the paper: the SSM's state-space
    matrices encode causal structure.
    """
    model.eval()
    K = X_tensor.shape[1]

    # Forward pass to populate B, C caches
    X_input = X_tensor[:500].unsqueeze(0).to(device)
    with torch.no_grad():
        _ = model(X_input)

    W_in = model.input_proj.weight.detach().cpu()   # [d_model, K]
    W_out = model.output_proj.weight.detach().cpu()  # [K, d_model]

    layer_bc_norms = []
    for layer in model.layers:
        if layer._B is None or layer._C is None:
            continue

        B = layer._B.float().cpu().mean(dim=(0, 1))  # [d_state]
        C = layer._C.float().cpu().mean(dim=(0, 1))  # [d_state]

        # Per-layer scalar: how much signal flows through this layer's SSM
        bc_strength = (C * B).sum().abs()
        layer_bc_norms.append(bc_strength)

        # Also get the layer's in_proj to understand routing
        # in_proj maps d_model -> 2*d_inner, first half is SSM input
        W_layer_in = layer.in_proj.weight.detach().cpu()  # [2*d_inner, d_model]
        d_inner = W_layer_in.shape[0] // 2
        W_ssm_in = W_layer_in[:d_inner]  # [d_inner, d_model]

        W_layer_out = layer.out_proj.weight.detach().cpu()  # [d_model, d_inner]

        # Per-layer variable-to-variable contribution through SSM
        # Path: W_in[d_model,K] -> W_ssm_in[d_inner,d_model] -> B·C -> W_layer_out[d_model,d_inner] -> W_out[K,d_model]
        # Approximate by: W_out @ W_layer_out @ W_ssm_in @ W_in * bc_strength
        layer_path = W_out @ W_layer_out @ W_ssm_in @ W_in  # [K, K]
        layer_bc_norms.append(layer_path.abs() * bc_strength)

    if not layer_bc_norms:
        return np.zeros((K, K))

    # Combine only the matrix contributions (skip scalar entries)
    matrix_scores = [s for s in layer_bc_norms if isinstance(s, torch.Tensor) and s.dim() == 2]
    if not matrix_scores:
        return extract_weight_score(model)

    S = torch.stack(matrix_scores).mean(0).numpy()
    np.fill_diagonal(S, 0)
    if S.max() > 0:
        S = S / S.max()
    return S


def run_core_experiment(
    K: int = 3,
    T: int = 1000,
    d_model: int = 32,
    d_state: int = 16,
    n_layers: int = 2,
    n_epochs: int = 80,
    device: str = "cpu",
    save_dir: str = "results/core_validation",
):
    """Run the core validation experiment."""

    os.makedirs(save_dir, exist_ok=True)
    print("=" * 70)
    print("CORE VALIDATION EXPERIMENT — Day 3 Test")
    print("=" * 70)

    # ── Step 1: Generate data ──
    print("\n[1/5] Generating VAR(1) data (chain: X0 -> X1 -> X2)...")
    X, A_true, G_true = generate_var1(K=K, T=T, graph_type="chain", seed=42)
    print(f"  Shape: ({T}, {K})")
    print(f"  True A:\n{A_true}")
    print(f"  Ground truth graph:\n{G_true}")

    # ── Step 2: Train ──
    print(f"\n[2/5] Training Mamba (d_model={d_model}, layers={n_layers}, epochs={n_epochs})...")
    X_tensor = torch.FloatTensor(X)

    model = MambaForTimeSeries(
        n_variables=K, d_model=d_model, d_state=d_state,
        n_layers=n_layers, dropout=0.05,
    )
    trainer = SSMTrainer(model, lr=1e-3, device=device)

    t0 = time.time()
    losses = trainer.train(X_tensor, n_epochs=n_epochs, batch_size=16,
                           seq_len=100, print_every=20)
    train_time = time.time() - t0
    print(f"  Time: {train_time:.1f}s | Final loss: {losses[-1]:.6f}")
    trainer.plot_loss(save_path=os.path.join(save_dir, "training_loss.png"))

    # ── Step 3: Extract causal scores (3 methods) ──
    print("\n[3/5] Extracting causal scores...")

    t0 = time.time()
    S_jacobian = extract_jacobian_score(model, X_tensor, device, seq_len=100)
    print(f"  Jacobian ({time.time()-t0:.1f}s):\n{np.array2string(S_jacobian, precision=3)}")

    t0 = time.time()
    S_weight = extract_weight_score(model)
    print(f"  Weight proj ({time.time()-t0:.3f}s):\n{np.array2string(S_weight, precision=3)}")

    t0 = time.time()
    S_bc = extract_bc_projected_score(model, X_tensor, device)
    print(f"  B·C^T proj ({time.time()-t0:.3f}s):\n{np.array2string(S_bc, precision=3)}")

    # ── Step 4: Baselines ──
    print("\n[4/5] Running baselines...")
    results_all = {}

    for name, S in [("SSM-Jacobian", S_jacobian), ("SSM-Weight", S_weight), ("SSM-BC", S_bc)]:
        m = compute_metrics(S, G_true)
        results_all[name] = m
        print(f"  {name} AUROC: {m['auroc']:.3f}")

    try:
        from baselines.granger import granger_causality_matrix
        t0 = time.time()
        S_granger = granger_causality_matrix(X, max_lag=5)
        m = compute_metrics(S_granger, G_true)
        m["runtime_s"] = time.time() - t0
        results_all["Granger"] = m
        print(f"  Granger AUROC: {m['auroc']:.3f} ({m['runtime_s']:.2f}s)")
    except ImportError:
        S_granger = None
        print("  [SKIP] Granger (statsmodels not found)")

    S_corr = np.abs(np.corrcoef(X.T))
    np.fill_diagonal(S_corr, 0)
    if S_corr.max() > 0:
        S_corr = S_corr / S_corr.max()
    m = compute_metrics(S_corr, G_true)
    results_all["Correlation"] = m
    print(f"  Correlation AUROC: {m['auroc']:.3f}")

    # ── Step 5: Visualize ──
    print("\n[5/5] Generating figures...")
    A_offdiag = np.abs(A_true.copy())
    np.fill_diagonal(A_offdiag, 0)
    if A_offdiag.max() > 0:
        A_offdiag = A_offdiag / A_offdiag.max()

    matrices = {"SSM-Jacobian": S_jacobian, "SSM-Weight": S_weight, "SSM-BC": S_bc}
    if S_granger is not None:
        matrices["Granger"] = S_granger
    matrices["Correlation"] = S_corr

    fig = plot_comparison(matrices, G_true=A_offdiag,
                          save_path=os.path.join(save_dir, "core_experiment.png"))
    plt.close(fig)

    print_metrics_table(results_all, dataset_name=f"VAR(1) Chain K={K} T={T}")

    # ── Decision ──
    best_ssm = max(
        results_all.get("SSM-Jacobian", {}).get("auroc", 0),
        results_all.get("SSM-Weight", {}).get("auroc", 0),
        results_all.get("SSM-BC", {}).get("auroc", 0),
    )

    print("\n" + "=" * 70)
    if best_ssm > 0.7:
        print(f"  PASS — Best SSM AUROC = {best_ssm:.3f} > 0.7")
        print("  THE PAPER IS REAL. Start the clock.")
    elif best_ssm > 0.5:
        print(f"  PARTIAL — Best SSM AUROC = {best_ssm:.3f}")
        print("  Signal exists but weak. Try: more epochs, larger d_state, different variant.")
    else:
        print(f"  FAIL — Best SSM AUROC = {best_ssm:.3f}")
        print("  No causal signal. See Risk 1 in plan: check convergence, extraction.")
    print("=" * 70)

    return results_all


def extract_causal_score(model, X_tensor, device="cpu", seq_len=100):
    """
    Unified extraction: returns (S_jacobian, S_weight) tuple.

    This is the main entry point used by exp1 and exp2.
    Returns the Jacobian-based score (gold standard) and
    the weight-projection score (fast approximation).
    """
    S_jac = extract_jacobian_score(model, X_tensor, device, seq_len=seq_len)
    S_weight = extract_weight_score(model)
    return S_jac, S_weight


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    run_core_experiment(device=device)
