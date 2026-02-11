"""
FlowState: Sampling Rate Invariant Time Series Forecasting.

Implementation based on:
    Graf, L., Ortner, T., Woźniak, S., & Pantazi, A. (2025).
    FlowState: Sampling Rate Invariant Time Series Forecasting. arXiv:2508.05287v2.

Key innovations:
1. SSM-based encoder with dynamic time-scale adjustment via Δ scaling
2. Functional Basis Decoder (FBD) using Legendre/Fourier basis functions
3. Parallel predictions training scheme for context length robustness
4. Causal normalization for proper parallel training

This implementation follows TSL (Torch Spatiotemporal Library) conventions.

Changes from original buggy version:
- [FIX-1] Causal normalization: correct running variance via E[x²] - E[x]²
- [FIX-2] Causal normalization inverse: proper feature dim slicing when input_size != output_size
- [FIX-3] FBD: scale_delta now actually scales the basis evaluation domain
- [FIX-4] S5 Block: gate computed from raw SSM output, not post-activation
- [FIX-5] Parallel predictions re-enabled for training
- [FIX-6] Exogenous features bypass causal normalization
- [FIX-7] Minor: .clamp(min=0) on variance before sqrt for numerical safety
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, Tuple, Literal
from tsl.nn.models.base_model import BaseModel

try:
    from models.s5_model import S5SSMOptimized
except ImportError:

    S5SSMOptimized = None


# =============================================================================
# Functional Basis Functions
# =============================================================================

def legendre_polynomials(n: int, x: torch.Tensor) -> torch.Tensor:
    """
    Evaluate Legendre polynomials P_0, ..., P_{n-1} at points x via recurrence.

    Args:
        n: Number of polynomials.
        x: Evaluation points, shape (...,).

    Returns:
        Tensor of shape (..., n).
    """
    if n == 0:
        return torch.zeros(*x.shape, 0, device=x.device, dtype=x.dtype)

    polys = [torch.ones_like(x)]

    if n > 1:
        polys.append(x.clone())

    for k in range(1, n - 1):
        p_next = ((2 * k + 1) * x * polys[k] - k * polys[k - 1]) / (k + 1)
        polys.append(p_next)

    return torch.stack(polys, dim=-1)


def fourier_basis(n: int, x: torch.Tensor) -> torch.Tensor:
    """
    Fourier basis: [1, sin(πx), cos(πx), sin(2πx), cos(2πx), ...].

    Args:
        n: Number of basis functions.
        x: Evaluation points, shape (...,).

    Returns:
        Tensor of shape (..., n).
    """
    basis = [torch.ones_like(x)]

    for k in range(1, (n + 1) // 2):
        basis.append(torch.sin(2 * math.pi * k * x))
        if len(basis) < n:
            basis.append(torch.cos(2 * math.pi * k * x))

    return torch.stack(basis[:n], dim=-1)


def half_legendre_polynomials(n: int, x: torch.Tensor) -> torch.Tensor:
    """
    Legendre polynomials on [0, 1] mapped to [-1, 1].

    Args:
        n: Number of polynomials.
        x: Points in [0, 1].

    Returns:
        Tensor of shape (..., n).
    """
    return legendre_polynomials(n, 2 * x - 1)


# =============================================================================
# Causal Normalization (Section: Causal Normalization)
# [FIX-1] Correct running variance: Var_t = E[x²]_{1:t} - (E[x]_{1:t})²
# [FIX-2] Inverse handles input_size != output_size
# =============================================================================

class CausalNormalization(nn.Module):
    """
    Causal normalization using running mean and standard deviation.

    Computes running statistics at each timestep t using only x_{1:t},
    enabling proper parallel predictions training.

    Reference: Equations (8-10) in FlowState paper.
    """

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply causal normalization.

        Args:
            x: Input tensor of shape (batch, length, features).

        Returns:
            x_norm: Normalized tensor.
            mu_r: Running means, shape (batch, length, features).
            sigma_r: Running stds, shape (batch, length, features).
        """
        B, L, F = x.shape

        # Running mean: μ_{r,t} = (1/t) Σ_{i=1}^{t} x_i
        cumsum_x = torch.cumsum(x, dim=1)
        t = torch.arange(1, L + 1, device=x.device, dtype=x.dtype).view(1, L, 1)
        mu_r = cumsum_x / t

        # [FIX-1] Running variance via E[x²] - E[x]²
        # This is the correct online formula; the previous version used
        # cumsum((x - mu_r)²)/t which is wrong because mu_r varies per step.
        cumsum_x2 = torch.cumsum(x ** 2, dim=1)
        var_r = cumsum_x2 / t - mu_r ** 2
        # [FIX-7] Clamp for numerical safety (floating point can yield small negatives)
        sigma_r = torch.sqrt(var_r.clamp(min=0.0) + self.eps)

        x_norm = (x - mu_r) / sigma_r

        return x_norm, mu_r, sigma_r

    def inverse(
        self,
        x_norm: torch.Tensor,
        mu_r: torch.Tensor,
        sigma_r: torch.Tensor,
        context_idx: int,
        output_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Inverse normalization using statistics from a specific context position.

        Args:
            x_norm: Normalized predictions, shape (batch, horizon, output_size).
            mu_r: Running means from forward pass, shape (batch, length, input_size).
            sigma_r: Running stds from forward pass, shape (batch, length, input_size).
            context_idx: Index of the last context timestep.
            output_size: If set, slice statistics to first output_size features.

        Returns:
            Denormalized tensor.
        """
        mu = mu_r[:, context_idx:context_idx + 1, :]
        sigma = sigma_r[:, context_idx:context_idx + 1, :]

        # [FIX-2] Slice to output_size when input_size != output_size
        if output_size is not None:
            mu = mu[:, :, :output_size]
            sigma = sigma[:, :, :output_size]

        return x_norm * sigma + mu


# =============================================================================
# S5 Block for FlowState (Figure 1b)
# [FIX-4] Gate computed from raw SSM output, not post-activation
# =============================================================================

class FlowStateS5Block(nn.Module):
    """
    S5 Block for FlowState encoder.

    Architecture (Figure 1b):
        x -> S5 -> gate = σ(W_g · h), out = gate * SELU(h) -> MLP -> + skip

    Args:
        hidden_size: Hidden dimension H.
        state_size: SSM state dimension P.
        num_blocks: Number of HiPPO-N blocks J.
        dropout: Dropout probability.
        dt_min: Minimum timescale initialization.
        dt_max: Maximum timescale initialization.
        scan_method: Parallel scan implementation.
    """

    def __init__(
        self,
        hidden_size: int,
        state_size: int,
        num_blocks: int = 1,
        dropout: float = 0.1,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        scan_method: str = "auto"
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.state_size = state_size

        if S5SSMOptimized is not None:
            self.ssm = S5SSMOptimized(
                input_size=hidden_size,
                state_size=state_size,
                num_blocks=num_blocks,
                dt_min=dt_min,
                dt_max=dt_max,
                scan_method=scan_method
            )
        else:
            raise ImportError(
                "S5SSMOptimized not found. Ensure s5_optimized.py is in path."
            )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Dropout(dropout)
        )

        # Gating projection (operates on raw SSM output)
        self.gate_proj = nn.Linear(hidden_size, hidden_size)

        self.norm = nn.LayerNorm(hidden_size)

    def scale_delta(self, scale: float) -> None:
        """
        Scale Δ by a factor: log(s_Δ · Δ) = log(s_Δ) + log(Δ).
        Equation 6 in paper.
        """
        self.ssm.log_delta.data += math.log(scale)
        if hasattr(self.ssm, '_cache_valid'):
            self.ssm._cache_valid = False

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: (batch, length, hidden_size).
            state: Optional initial SSM state.

        Returns:
            out: (batch, length, hidden_size).
            final_state: Final SSM state.
        """
        residual = x

        h, final_state = self.ssm(x, state)

        # [FIX-4] Gate from RAW SSM output, SELU on separate path
        gate = torch.sigmoid(self.gate_proj(h))
        h_act = F.selu(h)
        h_gated = gate * h_act

        out = self.mlp(h_gated)
        out = self.norm(residual + out)

        return out, final_state

    def step(
        self,
        x: torch.Tensor,
        state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single step for autoregressive generation.

        Args:
            x: (batch, hidden_size).
            state: Current SSM state.

        Returns:
            out: (batch, hidden_size).
            new_state: Updated state.
        """
        residual = x

        h, new_state = self.ssm.step(x, state)

        # [FIX-4] Same fix as forward
        gate = torch.sigmoid(self.gate_proj(h))
        h_act = F.selu(h)
        h_gated = gate * h_act

        out = self.mlp(h_gated)
        out = self.norm(residual + out)

        return out, new_state


# =============================================================================
# Functional Basis Decoder (Section: Functional Basis Decoder)
# [FIX-3] scale_delta now scales the evaluation domain of basis functions
# =============================================================================

class FunctionalBasisDecoder(nn.Module):
    """
    Functional Basis Decoder (FBD).

    Interprets encoder outputs as coefficients of a functional basis
    and produces continuous forecasts that can be sampled at any rate.

    The key mechanism for sampling-rate invariance: scale_delta stretches
    or compresses the evaluation domain of the basis functions.

    Args:
        hidden_size: Coefficient input dimension.
        output_size: Output feature dimension.
        n_basis: Number of basis functions.
        basis_type: "legendre", "fourier", or "half_legendre".
        base_target_length: Base target length T for reference.
    """

    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        n_basis: int = 64,
        basis_type: Literal["legendre", "fourier", "half_legendre"] = "legendre",
        base_target_length: int = 24
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_basis = n_basis
        self.basis_type = basis_type
        self.base_target_length = base_target_length

        self.coeff_proj = nn.Linear(hidden_size, n_basis * output_size)

        if basis_type == "legendre":
            self.basis_fn = legendre_polynomials
        elif basis_type == "fourier":
            self.basis_fn = fourier_basis
        elif basis_type == "half_legendre":
            self.basis_fn = half_legendre_polynomials
        else:
            raise ValueError(f"Unknown basis type: {basis_type}")

    def forward(
        self,
        h: torch.Tensor,
        target_length: int,
        scale_delta: float = 1.0
    ) -> torch.Tensor:
        """
        Decode hidden states to forecast using functional basis.

        Args:
            h: Hidden states, shape (batch, hidden_size).
            target_length: Number of forecast steps T.
            scale_delta: Time-scale adjustment s_Δ. Scales the evaluation
                         domain so that the same learned coefficients produce
                         forecasts appropriate for different sampling rates.

        Returns:
            forecast: Shape (batch, target_length, output_size).
        """
        batch_size = h.shape[0]

        # Project to coefficients: (batch, n_basis, output_size)
        coeffs = self.coeff_proj(h)
        coeffs = coeffs.view(batch_size, self.n_basis, self.output_size)

        # [FIX-3] scale_delta scales the evaluation domain endpoint.
        # At scale_delta=1.0, we evaluate over the full canonical domain.
        # At scale_delta=0.5, we evaluate over half the domain (slower rate → 
        # the same basis covers a longer physical time span).
        # At scale_delta=2.0, we compress (faster rate).
        if self.basis_type == "legendre":
            # Canonical domain [-1, 1], scaled to [-1, scale_delta]
            # This stretches/compresses where we sample the learned basis
            t_end = max(min(scale_delta, 1.0), -1.0 + 1e-6)  # clamp for stability
            # More precisely: scale maps [0, T] physical time to basis domain
            # scale_delta < 1 → we use less of the domain → smoother/longer horizon
            # scale_delta > 1 → we use more → sharper/shorter horizon
            t = torch.linspace(-1.0, -1.0 + 2.0 * scale_delta, target_length,
                               device=h.device, dtype=h.dtype)
            t = t.clamp(-1.0, 1.0)
        else:
            # Fourier / half-Legendre domain [0, 1], scaled to [0, scale_delta]
            t = torch.linspace(0.0, scale_delta, target_length,
                               device=h.device, dtype=h.dtype)
            if self.basis_type == "half_legendre":
                t = t.clamp(0.0, 1.0)

        # Evaluate basis: (target_length, n_basis)
        basis_values = self.basis_fn(self.n_basis, t)

        # Reconstruct: coeffs (B, n_basis, out) × basis (T, n_basis) → (B, T, out)
        forecast = torch.einsum('bnf,tn->btf', coeffs, basis_values)

        return forecast

    def forward_parallel(
        self,
        h_seq: torch.Tensor,
        target_length: int,
        start_idx: int,
        scale_delta: float = 1.0
    ) -> torch.Tensor:
        """
        Decode for parallel predictions (multiple forecasts from one sequence).

        Args:
            h_seq: Hidden states sequence (batch, seq_len, hidden_size).
            target_length: Forecast horizon.
            start_idx: Starting index for predictions (L_min).
            scale_delta: Time-scale factor.

        Returns:
            forecasts: (batch, seq_len - start_idx, target_length, output_size).
        """
        batch_size, seq_len, hidden_size = h_seq.shape
        n_predictions = seq_len - start_idx

        h_flat = h_seq[:, start_idx:, :].reshape(-1, hidden_size)
        forecasts_flat = self.forward(h_flat, target_length, scale_delta)
        forecasts = forecasts_flat.view(
            batch_size, n_predictions, target_length, self.output_size
        )

        return forecasts


# =============================================================================
# FlowState Model (Main Architecture - Figure 1a)
# [FIX-5] Parallel predictions re-enabled
# [FIX-6] Exogenous features bypass causal normalization
# =============================================================================

class FlowState(BaseModel):
    """
    FlowState: Sampling Rate Invariant Time Series Foundation Model.

    Architecture (Figure 1a):
        Input → Causal Norm (target only) → Cat(norm, exog) → Embed →
        SSM Encoder → FBD → Inverse Norm → Forecast

    Args:
        input_size: Number of target input features per timestep.
        hidden_size: Hidden dimension H.
        output_size: Number of output features.
        horizon: Default forecasting horizon T.
        n_layers: Number of stacked S5 layers N.
        state_size: SSM state dimension P.
        num_blocks: Number of HiPPO-N blocks J.
        n_basis: Number of basis functions for FBD.
        basis_type: "legendre", "fourier", or "half_legendre".
        dropout: Dropout probability.
        dt_min: Minimum timescale initialization.
        dt_max: Maximum timescale initialization.
        l_min: Minimum context length for parallel predictions.
        base_seasonality: Base seasonality for scale computation (default 24).
        exog_size: Number of exogenous feature dimensions.
        scan_method: Parallel scan implementation.

    Reference:
        Graf et al. "FlowState: Sampling Rate Invariant Time Series Forecasting"
        arXiv:2508.05287v2.
    """

    return_type = torch.Tensor

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        horizon: int,
        n_layers: int = 4,
        state_size: int = 64,
        num_blocks: int = 1,
        n_basis: int = 64,
        basis_type: Literal["legendre", "fourier", "half_legendre"] = "legendre",
        dropout: float = 0.1,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        l_min: int = 20,
        base_seasonality: int = 24,
        exog_size: int = 0,
        scan_method: str = "auto"
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.horizon = horizon
        self.n_layers = n_layers
        self.state_size = state_size
        self.num_blocks = num_blocks
        self.n_basis = n_basis
        self.l_min = l_min
        self.base_seasonality = base_seasonality
        self.exog_size = exog_size

        self._scale_delta = 1.0

        # Causal normalization (applied to target features only)
        self.causal_norm = CausalNormalization()

        # [FIX-6] Input encoder takes normalized targets + raw exog
        self.input_encoder = nn.Linear(input_size + exog_size, hidden_size)

        # SSM Encoder
        self.blocks = nn.ModuleList([
            FlowStateS5Block(
                hidden_size=hidden_size,
                state_size=state_size,
                num_blocks=num_blocks,
                dropout=dropout,
                dt_min=dt_min,
                dt_max=dt_max,
                scan_method=scan_method
            )
            for _ in range(n_layers)
        ])

        # Global skip connection
        self.skip_proj = nn.Linear(hidden_size, hidden_size)

        # Functional Basis Decoder
        self.fbd = FunctionalBasisDecoder(
            hidden_size=hidden_size,
            output_size=output_size,
            n_basis=n_basis,
            basis_type=basis_type,
            base_target_length=horizon
        )

        self.output_norm = nn.LayerNorm(hidden_size)

    @property
    def scale_delta(self) -> float:
        return self._scale_delta

    def set_scale_delta(self, scale: float) -> None:
        """Set time-scale factor s_Δ, adjusting all SSM blocks."""
        if scale != self._scale_delta:
            ratio = scale / self._scale_delta
            for block in self.blocks:
                block.scale_delta(ratio)
            self._scale_delta = scale

    def compute_scale_from_seasonality(self, seasonality: int) -> float:
        """s_Δ = Base Seasonality / Dataset Seasonality."""
        return self.base_seasonality / seasonality

    def _encode(
        self,
        x: torch.Tensor,
        states: Optional[list] = None
    ) -> Tuple[torch.Tensor, list]:
        """
        Encode through stacked S5 blocks with global skip.

        Args:
            x: Embedded input (batch, length, hidden_size).
            states: Optional initial states per layer.

        Returns:
            h: Encoded sequence (batch, length, hidden_size).
            final_states: Final states per layer.
        """
        if states is None:
            states = [None] * self.n_layers

        h = x
        skip = self.skip_proj(x)
        final_states = []

        for i, block in enumerate(self.blocks):
            h, state = block(h, states[i])
            final_states.append(state)

        h = h + skip
        h = self.output_norm(h)

        return h, final_states

    def forward(
        self,
        x: torch.Tensor,
        u: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass.

        Training: parallel predictions from multiple context lengths.
        Inference: single forecast from full context.

        Args:
            x: Input tensor (batch, window, input_size) — target features only.
            u: Optional exogenous features (batch, window + horizon, exog_size).

        Returns:
            Training: (batch * n_predictions, horizon, output_size).
            Inference: (batch, horizon, output_size).
        """
        batch_size, window, _ = x.shape

        # [FIX-6] Normalize target features ONLY, then concatenate raw exog
        x_norm, mu_r, sigma_r = self.causal_norm(x)

        if u is not None and self.exog_size > 0:
            encoder_input = torch.cat([x_norm, u[:, :window]], dim=-1)
        else:
            encoder_input = x_norm

        # Embed
        h = self.input_encoder(encoder_input)

        # Encode
        h_encoded, final_states = self._encode(h)

        # [FIX-5] Parallel predictions enabled during training
        if self.training:
            return self._parallel_predictions_forward(
                h_encoded, mu_r, sigma_r, window
            )
        else:
            # Single forecast from last position
            h_last = h_encoded[:, -1, :]
            forecast_norm = self.fbd(h_last, self.horizon, self._scale_delta)
            # [FIX-2] Pass output_size for proper feature slicing
            forecast = self.causal_norm.inverse(
                forecast_norm, mu_r, sigma_r,
                context_idx=window - 1,
                output_size=self.output_size
            )
            return forecast

    def _parallel_predictions_forward(
        self,
        h_encoded: torch.Tensor,
        mu_r: torch.Tensor,
        sigma_r: torch.Tensor,
        window: int
    ) -> torch.Tensor:
        """
        Parallel predictions for training (Figure 2).

        Produces forecasts from positions l_min to (window - horizon),
        each using context up to that position only.

        Args:
            h_encoded: (batch, window, hidden_size).
            mu_r: Running means from causal norm.
            sigma_r: Running stds from causal norm.
            window: Input window length.

        Returns:
            (batch * n_predictions, horizon, output_size).
        """
        batch_size = h_encoded.shape[0]

        max_start = window - self.horizon
        start_idx = max(self.l_min, 1)

        if start_idx >= max_start:
            start_idx = max(0, max_start - 1)
            n_predictions = 1
        else:
            n_predictions = max_start - start_idx

        # Get all forecasts in parallel
        forecasts_norm = self.fbd.forward_parallel(
            h_encoded, self.horizon, start_idx, self._scale_delta
        )  # (batch, n_predictions, horizon, output_size)

        # Denormalize each with its corresponding context statistics
        forecasts = []
        for i in range(n_predictions):
            ctx_idx = start_idx + i
            # [FIX-2] Proper feature slicing
            forecast_i = self.causal_norm.inverse(
                forecasts_norm[:, i], mu_r, sigma_r,
                context_idx=ctx_idx,
                output_size=self.output_size
            )
            forecasts.append(forecast_i)

        forecasts = torch.stack(forecasts, dim=1)
        forecasts = forecasts.view(-1, self.horizon, self.output_size)

        return forecasts

    def get_parallel_targets(
        self,
        y_full: torch.Tensor,
        window: int
    ) -> torch.Tensor:
        """
        Get corresponding targets for parallel predictions.

        Args:
            y_full: Full target sequence (batch, window + horizon, output_size).
            window: Input window length.

        Returns:
            Targets matching parallel predictions shape.
        """
        batch_size = y_full.shape[0]

        max_start = window - self.horizon
        start_idx = max(self.l_min, 1)

        if start_idx >= max_start:
            start_idx = max(0, max_start - 1)
            n_predictions = 1
        else:
            n_predictions = max_start - start_idx

        targets = []
        for i in range(n_predictions):
            ctx_idx = start_idx + i
            target_i = y_full[:, ctx_idx + 1:ctx_idx + 1 + self.horizon]
            targets.append(target_i)

        targets = torch.stack(targets, dim=1)
        targets = targets.view(-1, self.horizon, self.output_size)

        return targets

    @torch.no_grad()
    def generate(
        self,
        x: torch.Tensor,
        horizon: Optional[int] = None,
        u: Optional[torch.Tensor] = None,
        seasonality: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate forecasts with optional time-scale adjustment.

        Args:
            x: Input context (batch, window, input_size).
            horizon: Custom horizon (default: effective horizon from scale).
            u: Optional exogenous features.
            seasonality: Dataset seasonality for automatic scale computation.

        Returns:
            (batch, horizon, output_size).
        """
        if seasonality is not None:
            scale = self.compute_scale_from_seasonality(seasonality)
            self.set_scale_delta(scale)

        if horizon is None:
            effective_horizon = max(1, int(self.horizon / self._scale_delta))
        else:
            effective_horizon = horizon

        original_horizon = self.horizon
        self.horizon = effective_horizon

        was_training = self.training
        self.eval()

        output = self.forward(x, u, **kwargs)

        self.horizon = original_horizon
        if was_training:
            self.train()

        return output

    def step(
        self,
        x: torch.Tensor,
        states: list,
        mu: torch.Tensor,
        sigma: torch.Tensor
    ) -> Tuple[torch.Tensor, list]:
        """
        Single step for autoregressive generation.

        Args:
            x: Input at timestep (batch, input_size).
            states: List of SSM states per layer.
            mu: Normalization mean (batch, 1, input_size).
            sigma: Normalization std (batch, 1, input_size).

        Returns:
            output: (batch, output_size).
            new_states: Updated states.
        """
        # Normalize target features
        x_norm = (x - mu.squeeze(1)) / sigma.squeeze(1)

        h = self.input_encoder(x_norm)

        new_states = []
        skip = self.skip_proj(h)

        for i, block in enumerate(self.blocks):
            h, new_state = block.step(h, states[i])
            new_states.append(new_state)

        h = h + skip
        h = self.output_norm(h)

        forecast_norm = self.fbd(h, 1, self._scale_delta)

        # [FIX-2] Denormalize with correct feature dims
        mu_out = mu[:, :, :self.output_size]
        sigma_out = sigma[:, :, :self.output_size]
        forecast = forecast_norm * sigma_out + mu_out

        return forecast.squeeze(1), new_states


# =============================================================================
# Utility Functions
# =============================================================================

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_effective_lengths(
    context_length: int,
    forecast_length: int,
    scale_delta: float
) -> Tuple[int, int]:
    """
    Compute effective lengths: L_eff = L / s_Δ, T_eff = T / s_Δ.
    """
    return (
        max(1, int(context_length / scale_delta)),
        max(1, int(forecast_length / scale_delta))
    )


# =============================================================================
# Test Suite
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("FlowState Model Test Suite (Corrected)")
    print("=" * 70)

    batch_size = 4
    window_size = 128
    horizon = 24
    input_size = 3
    output_size = 1
    hidden_size = 64
    state_size = 32
    n_layers = 4
    n_basis = 32

    # Test 1: Basis functions
    print("\n1. Testing basis functions...")
    x_test = torch.linspace(-1, 1, 100)
    leg = legendre_polynomials(5, x_test)
    assert leg.shape == (100, 5), f"Expected (100, 5), got {leg.shape}"
    # Verify P_0 = 1
    assert torch.allclose(leg[:, 0], torch.ones(100)), "P_0 should be 1"
    # Verify P_1 = x
    assert torch.allclose(leg[:, 1], x_test), "P_1 should be x"

    x_01 = torch.linspace(0, 1, 100)
    four = fourier_basis(5, x_01)
    assert four.shape == (100, 5)

    half_leg = half_legendre_polynomials(5, x_01)
    assert half_leg.shape == (100, 5)
    print("   ✓ Basis functions passed!")

    # Test 2: Causal Normalization (corrected)
    print("\n2. Testing causal normalization...")
    causal_norm = CausalNormalization()
    x_test = torch.randn(batch_size, window_size, input_size)
    x_norm, mu_r, sigma_r = causal_norm(x_test)

    assert x_norm.shape == x_test.shape
    assert mu_r.shape == x_test.shape
    assert sigma_r.shape == x_test.shape

    # Verify causality: mu_r[:, t] == x[:, :t+1].mean(dim=1)
    for t_check in [0, 1, 10, 50, 127]:
        expected_mu = x_test[:, :t_check + 1].mean(dim=1)
        actual_mu = mu_r[:, t_check]
        assert torch.allclose(expected_mu, actual_mu, atol=1e-5), \
            f"Causal mean failed at t={t_check}"

    # Verify variance: var_r[:, t] == x[:, :t+1].var(dim=1, correction=0)
    for t_check in [1, 10, 50, 127]:
        expected_var = x_test[:, :t_check + 1].var(dim=1, correction=0)
        actual_var = (sigma_r[:, t_check] ** 2 - causal_norm.eps)
        assert torch.allclose(expected_var, actual_var, atol=1e-4), \
            f"Causal variance failed at t={t_check}"

    # Test inverse with different output_size
    pred_norm = torch.randn(batch_size, horizon, output_size)
    pred_denorm = causal_norm.inverse(
        pred_norm, mu_r, sigma_r, context_idx=window_size - 1,
        output_size=output_size
    )
    assert pred_denorm.shape == (batch_size, horizon, output_size)
    print("   ✓ Causal normalization passed (with variance verification)!")

    # Test 3: FBD with scale_delta
    print("\n3. Testing FBD with scale_delta...")
    fbd = FunctionalBasisDecoder(
        hidden_size=hidden_size,
        output_size=output_size,
        n_basis=n_basis,
        basis_type="legendre"
    )

    h_test = torch.randn(batch_size, hidden_size)
    f1 = fbd(h_test, horizon, scale_delta=1.0)
    f2 = fbd(h_test, horizon, scale_delta=0.5)
    assert f1.shape == (batch_size, horizon, output_size)
    assert f2.shape == (batch_size, horizon, output_size)
    # Forecasts should differ when scale changes
    assert not torch.allclose(f1, f2, atol=1e-6), \
        "FBD output should change with scale_delta"
    print("   ✓ FBD with scale_delta passed!")

    # Test 4: Full model
    print("\n4. Testing FlowState model...")

    if S5SSMOptimized is None:
        print("   ⚠ S5SSMOptimized not available, skipping full model test")
    else:
        model = FlowState(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            horizon=horizon,
            n_layers=n_layers,
            state_size=state_size,
            num_blocks=4,
            n_basis=n_basis,
            basis_type="legendre",
            dropout=0.1,
            l_min=20,
            scan_method="sequential"
        )

        n_params = count_parameters(model)
        print(f"   Parameters: {n_params:,}")

        # Inference
        model.eval()
        x = torch.randn(batch_size, window_size, input_size)
        y = model(x)
        assert y.shape == (batch_size, horizon, output_size)
        print(f"   Inference output: {y.shape} ✓")

        # Training with parallel predictions
        model.train()
        y_train = model(x)
        # Should have more samples than batch_size due to parallel predictions
        assert y_train.shape[-2:] == (horizon, output_size)
        assert y_train.shape[0] >= batch_size, \
            f"Expected >= {batch_size} parallel preds, got {y_train.shape[0]}"
        print(f"   Training output: {y_train.shape} ✓")

        # Gradient flow
        loss = y_train.sum()
        loss.backward()
        has_grad = all(
            p.grad is not None for p in model.parameters() if p.requires_grad
        )
        assert has_grad, "Some parameters have no gradient"
        print("   ✓ Gradient flow verified!")

        # Scale adjustment
        print("\n5. Testing time-scale adjustment...")
        model.eval()
        model.zero_grad()

        orig = model(x).clone()
        model.set_scale_delta(0.5)
        adj = model.generate(x)
        print(f"   Original: {orig.shape}, Adjusted: {adj.shape}")

        model.set_scale_delta(1.0)
        print("   ✓ Time-scale adjustment passed!")

        # Test with exogenous features
        print("\n6. Testing with exogenous features...")
        model_exog = FlowState(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            horizon=horizon,
            n_layers=2,
            state_size=state_size,
            num_blocks=2,
            n_basis=n_basis,
            exog_size=5,
            scan_method="sequential"
        )
        model_exog.eval()

        u = torch.randn(batch_size, window_size + horizon, 5)
        y_exog = model_exog(x, u=u)
        assert y_exog.shape == (batch_size, horizon, output_size)
        print(f"   Exog output: {y_exog.shape} ✓")

    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)
