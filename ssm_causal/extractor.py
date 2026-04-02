"""
SSM Causal Extractor — Extract B and C matrices from trained Mamba models
and compute the causal score matrix S = C · B^T.

Supports:
    - Mamba-1 / Mamba-2 architectures (via mamba_ssm library)
    - Custom SSM models (via manual B/C registration)
    - Three score variants: time_avg, gate_weighted, layer_agg
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Dict


class SSMCausalExtractor:
    """
    Extracts B and C matrices from a trained Mamba model
    and computes the SSM causal score matrix.

    Usage:
        extractor = SSMCausalExtractor(model)
        S = extractor.compute_score(X, variant='time_avg')
        # S[i, j] represents the causal influence of variable j on variable i
    """

    def __init__(
        self,
        model: nn.Module,
        layer_indices: Optional[List[int]] = None,
        hook_target: str = "auto",
    ):
        """
        Args:
            model: A trained Mamba model.
            layer_indices: Which SSM layers to extract from. None = all layers.
            hook_target: How to find SSM layers.
                "auto" — search for known Mamba layer names.
                "manual" — user must call register_custom_hooks().
        """
        self.model = model
        self.layer_indices = layer_indices
        self.hook_target = hook_target

        self._B_cache: List[torch.Tensor] = []
        self._C_cache: List[torch.Tensor] = []
        self._gate_cache: List[torch.Tensor] = []
        self._hooks: List[torch.utils.hooks.RemovableHook] = []
        self._ssm_layers: List[nn.Module] = []

        if hook_target == "auto":
            self._find_ssm_layers()

    def _find_ssm_layers(self):
        """Locate SSM layers in the model by searching for known attribute patterns."""
        self._ssm_layers = []
        for name, module in self.model.named_modules():
            # Check for common Mamba layer patterns
            if any(
                hasattr(module, attr)
                for attr in ["B", "in_proj", "x_proj", "dt_proj"]
            ):
                if hasattr(module, "forward"):
                    self._ssm_layers.append(module)

        # If nothing found, try to find layers from the backbone
        if not self._ssm_layers:
            for name, module in self.model.named_modules():
                class_name = module.__class__.__name__.lower()
                if "mamba" in class_name or "ssm" in class_name:
                    self._ssm_layers.append(module)

    def register_hooks(self):
        """Attach forward hooks to capture B, C, and gate values."""
        self._clear_caches()

        layers = self._ssm_layers
        if self.layer_indices is not None:
            layers = [layers[i] for i in self.layer_indices if i < len(layers)]

        for layer in layers:
            hook = layer.register_forward_hook(self._bc_hook)
            self._hooks.append(hook)

    def _bc_hook(self, module, input, output):
        """
        Capture B, C matrices during forward pass.

        Mamba's selective SSM computes B and C as data-dependent projections.
        They are typically stored as intermediate tensors during forward().
        We access them via attributes set during the forward pass.
        """
        # Try multiple access patterns for different Mamba versions
        B, C = None, None

        # Pattern 1: Direct attributes (some Mamba implementations)
        if hasattr(module, "_B") and module._B is not None:
            B = module._B.detach().cpu()
        if hasattr(module, "_C") and module._C is not None:
            C = module._C.detach().cpu()

        # Pattern 2: Stored in ssm_state
        if B is None and hasattr(module, "B"):
            if isinstance(module.B, torch.Tensor):
                B = module.B.detach().cpu()

        if C is None and hasattr(module, "C"):
            if isinstance(module.C, torch.Tensor):
                C = module.C.detach().cpu()

        if B is not None:
            self._B_cache.append(B)
        if C is not None:
            self._C_cache.append(C)

        # Capture gate activations if available
        if hasattr(module, "_gate") and module._gate is not None:
            self._gate_cache.append(module._gate.detach().cpu())

    def remove_hooks(self):
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def _clear_caches(self):
        """Clear cached B, C, gate tensors."""
        self._B_cache.clear()
        self._C_cache.clear()
        self._gate_cache.clear()

    def compute_score(
        self,
        X: torch.Tensor,
        variant: str = "time_avg",
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Compute SSM causal score matrix.

        Args:
            X: Input time series [batch, seq_len, channels].
            variant: Score computation method:
                'time_avg' — Average B·C^T over timesteps (simplest).
                'gate_weighted' — Weight by gate activation magnitude.
                'layer_agg' — Aggregate across all SSM layers.
            normalize: If True, normalize scores to [0, 1].

        Returns:
            S: Causal score matrix [channels, channels].
               S[i, j] > 0 suggests variable j causes variable i.
        """
        self._clear_caches()
        self.register_hooks()

        with torch.no_grad():
            _ = self.model(X)

        self.remove_hooks()

        if variant == "time_avg":
            S = self._time_averaged_score()
        elif variant == "gate_weighted":
            S = self._gate_weighted_score()
        elif variant == "layer_agg":
            S = self._layer_aggregated_score()
        else:
            raise ValueError(f"Unknown variant: {variant}")

        if normalize and S.abs().max() > 0:
            S = S / S.abs().max()

        return S

    def _time_averaged_score(self) -> torch.Tensor:
        """
        Variant A: Simple time-averaged B·C^T.

        Average B and C across timesteps to get single matrices,
        then compute the product.
        """
        if not self._B_cache or not self._C_cache:
            raise RuntimeError(
                "No B/C matrices captured. Check that hooks are targeting "
                "the correct layers and that the model has SSM components."
            )

        # Stack across layers and average
        B = torch.stack(self._B_cache).mean(0)  # average over layers
        C = torch.stack(self._C_cache).mean(0)

        # Handle different tensor shapes:
        # B might be [batch, seq, state_dim, K] or [batch, seq, state_dim]
        # We need to reduce to [state_dim, K] and [K, state_dim]

        B = self._reduce_to_2d(B, "B")  # -> [state_dim, K]
        C = self._reduce_to_2d(C, "C")  # -> [K, state_dim]

        return (C @ B.T).abs()  # [K, K]

    def _gate_weighted_score(self) -> torch.Tensor:
        """
        Variant B: Weight each timestep's B·C^T by gate activation magnitude.

        High gate activation = model is actively processing this input.
        Low gate activation = model is suppressing this input.
        Weighting by gate gives a more selective causal signal.
        """
        if not self._gate_cache:
            # Fall back to time_avg if no gate information
            return self._time_averaged_score()

        B = torch.stack(self._B_cache).mean(0)
        C = torch.stack(self._C_cache).mean(0)
        gates = torch.stack(self._gate_cache).mean(0)

        B = self._reduce_to_2d(B, "B")
        C = self._reduce_to_2d(C, "C")

        # Gate weighting: scale the score by average gate magnitude
        gate_weight = gates.abs().mean()
        return (C @ B.T).abs() * gate_weight

    def _layer_aggregated_score(self) -> torch.Tensor:
        """
        Variant C: Aggregate B·C^T across all layers.

        Different layers capture different timescales:
            - Early layers: short-range (small lag) causal effects
            - Deep layers: long-range (large lag) causal effects
        Aggregating gives a multi-scale causal view.
        """
        if len(self._B_cache) != len(self._C_cache):
            raise RuntimeError("Mismatched B/C cache sizes across layers")

        scores = []
        for B_l, C_l in zip(self._B_cache, self._C_cache):
            B_2d = self._reduce_to_2d(B_l, "B")
            C_2d = self._reduce_to_2d(C_l, "C")
            scores.append((C_2d @ B_2d.T).abs())

        return torch.stack(scores).mean(0)  # [K, K]

    @staticmethod
    def _reduce_to_2d(tensor: torch.Tensor, name: str) -> torch.Tensor:
        """Reduce a multi-dimensional B or C tensor to 2D by averaging extra dims."""
        while tensor.dim() > 2:
            tensor = tensor.mean(0)
        return tensor

    def get_layer_scores(self, X: torch.Tensor) -> List[torch.Tensor]:
        """
        Get per-layer causal score matrices (useful for analysis).

        Returns a list of [K, K] score matrices, one per SSM layer.
        """
        self._clear_caches()
        self.register_hooks()

        with torch.no_grad():
            _ = self.model(X)

        self.remove_hooks()

        scores = []
        for B_l, C_l in zip(self._B_cache, self._C_cache):
            B_2d = self._reduce_to_2d(B_l, "B")
            C_2d = self._reduce_to_2d(C_l, "C")
            S = (C_2d @ B_2d.T).abs()
            if S.abs().max() > 0:
                S = S / S.abs().max()
            scores.append(S)

        return scores


class ManualSSMExtractor:
    """
    For custom SSM models where B and C are directly accessible
    (not through hooks). Use this when you have direct access to
    the model's B, C parameters.

    Usage:
        extractor = ManualSSMExtractor()
        S = extractor.compute_from_matrices(B, C)
    """

    @staticmethod
    def compute_from_matrices(
        B: torch.Tensor,
        C: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Compute causal score directly from B and C matrices.

        Args:
            B: Input-to-state matrix [state_dim, K] or [batch, seq, state_dim, K].
            C: State-to-output matrix [K, state_dim] or [batch, seq, K, state_dim].
            normalize: Normalize to [0, 1].

        Returns:
            S: Causal score matrix [K, K].
        """
        while B.dim() > 2:
            B = B.mean(0)
        while C.dim() > 2:
            C = C.mean(0)

        S = (C @ B.T).abs()

        if normalize and S.abs().max() > 0:
            S = S / S.abs().max()

        return S
