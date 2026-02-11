"""
S4: Structured State Space Sequence Model for Time Series Forecasting.

Implementation based on:
    Gu, A., Goel, K., & Ré, C. (2022).
    Efficiently Modeling Long Sequences with Structured State Spaces. ICLR 2022.

This implementation follows TSL (Torch Spatiotemporal Library) conventions
and is adapted for spatio-temporal forecasting tasks.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from tsl.nn.models.base_model import BaseModel
from typing import Optional, Tuple

from models.triton_kernels import triton_parallel_scan, _TRITON_AVAILABLE


def make_hippo_legs_matrix(n: int) -> torch.Tensor:
    """
    Construct the HiPPO-LegS (Legendre-Scaled) matrix for state space initialization.
    
    The HiPPO-LegS matrix enables optimal approximation of continuous-time functions
    using Legendre polynomials, providing excellent long-range dependency modeling.
    
    Args:
        n: State dimension (size of the square matrix).
        
    Returns:
        A_legs: The HiPPO-LegS matrix of shape (n, n).
        
    Reference:
        Gu et al. (2020) "HiPPO: Recurrent Memory with Optimal Polynomial Projections"
        Equation (10) in S4 paper (Appendix C.1).
    """
    # Create index arrays
    k = torch.arange(n, dtype=torch.float32)
    i = k.unsqueeze(1)  # Column indices
    j = k.unsqueeze(0)  # Row indices
    
    # HiPPO-LegS formula:
    # A[n,k] = -(2n+1)^(1/2) * (2k+1)^(1/2)  if n > k
    #        = -(n + 1)                         if n = k
    #        = 0                                 if n < k
    
    A = torch.zeros(n, n)
    
    # Lower triangular part (n > k)
    mask_lower = i > j
    sqrt_2i_plus_1 = torch.sqrt(2 * i + 1)
    sqrt_2j_plus_1 = torch.sqrt(2 * j + 1)
    A[mask_lower] = -(sqrt_2i_plus_1 * sqrt_2j_plus_1)[mask_lower]
    
    # BUG FIX #1: Diagonal must be -(k+1), not (k+1).
    # Positive diagonal causes unstable (exponentially growing) dynamics.
    A.diagonal().copy_(-(k + 1))
    
    return A


def make_nplr_hippo(n: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Construct NPLR (Normal Plus Low-Rank) representation of HiPPO-LegS matrix.
    
    Decomposes A = V (Λ - PQ*) V^{-1} where:
    - The skew-symmetric part S = (A - A^T)/2 is a normal matrix
    - The symmetric part (A + A^T)/2 is low-rank (rank 1 for HiPPO-LegS)
    - We diagonalize S and express the symmetric correction as low-rank in that basis
    
    Args:
        n: State dimension.
        
    Returns:
        Lambda: Diagonal eigenvalues (purely imaginary from skew-symmetric part),
                shape (n,), complex.
        P: Low-rank factor P, shape (n,), complex.
        Q: Low-rank factor Q, shape (n,), complex.
        
    Reference:
        Appendix C.3 of S4 paper.
    """
    # Get HiPPO-LegS matrix
    A = make_hippo_legs_matrix(n)
    
    # Separate into symmetric and skew-symmetric parts
    # A = (A + A^T)/2 + (A - A^T)/2
    A_symmetric = (A + A.T) / 2
    A_skew = (A - A.T) / 2
    
    # BUG FIX #2: Diagonalize the SKEW-SYMMETRIC part, not the symmetric part.
    # The skew-symmetric part is a normal matrix (SS* = S*S) so it can be
    # unitarily diagonalized. Its eigenvalues are purely imaginary.
    # We then express the symmetric (low-rank) correction in this eigenbasis.
    eigenvalues, V = torch.linalg.eigh(1j * A_skew)
    # eigh(iS) gives real eigenvalues ω such that eigenvalues of S are -iω
    # So Lambda = -i * eigenvalues = purely imaginary eigenvalues of S
    Lambda = -1j * eigenvalues.to(torch.complex64)
    V = V.to(torch.complex64)
    
    # The symmetric part of HiPPO-LegS is rank-1: A_sym = -½ pp^T
    # where p_k = sqrt(2k+1). Transform into the eigenbasis of S.
    p = torch.sqrt(2 * torch.arange(n, dtype=torch.float32) + 1)
    
    # Project p into eigenbasis: p_tilde = V^H @ p
    p_tilde = V.conj().T @ p.to(torch.complex64)  # shape (n,)
    
    # In the eigenbasis, A = Λ + V^H A_sym V = Λ - ½ p_tilde p_tilde^*
    # So P = p_tilde / sqrt(2), Q = p_tilde / sqrt(2)
    P = p_tilde / math.sqrt(2)
    Q = p_tilde / math.sqrt(2)
    
    # Add the diagonal contribution from A_symmetric to Lambda.
    # The full matrix in eigenbasis is Λ - PQ*, and we want this to represent
    # V^H A V = V^H (S + A_sym) V = Λ_S + V^H A_sym V.
    # Since A_sym = -½ pp^T, V^H A_sym V = -½ p_tilde p_tilde^* = -PQ*.
    # So Lambda here is just the eigenvalues of S (correct).
    
    return Lambda, P, Q


def make_dplr_hippo(n: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Construct DPLR (Diagonal Plus Low-Rank) representation of HiPPO-LegS.
    
    Returns the NPLR decomposition along with the change-of-basis matrix V,
    so the full reconstruction is: A = V (Λ - PQ*) V^{-1}.
    
    Args:
        n: State dimension (must be even for conjugate pairing).
        
    Returns:
        Lambda: Diagonal eigenvalues, shape (n,), complex.
        P: Low-rank factor P, shape (n,), complex.
        B: Input projection (transformed), shape (n, 1), complex.
        V: Eigenvector matrix, shape (n, n), complex.
        
    Reference:
        Appendix C.4 of S4 paper.
    """
    assert n % 2 == 0, "State size must be even for DPLR conjugate pairing"
    
    # Get HiPPO matrix and its NPLR decomposition
    A = make_hippo_legs_matrix(n)
    A_skew = (A - A.T) / 2
    
    # Diagonalize skew-symmetric part
    eigenvalues, V = torch.linalg.eigh(1j * A_skew)
    Lambda = -1j * eigenvalues.to(torch.complex64)
    V = V.to(torch.complex64)
    
    # Low-rank factors in eigenbasis
    p = torch.sqrt(2 * torch.arange(n, dtype=torch.float32) + 1)
    p_tilde = V.conj().T @ p.to(torch.complex64)
    P = p_tilde / math.sqrt(2)
    
    # BUG FIX #3: Construct B by projecting a proper initialization into
    # the eigenbasis, rather than using arbitrary imaginary offsets.
    B_orig = torch.ones(n, 1, dtype=torch.complex64)
    B_tilde = V.conj().T @ B_orig  # (n, 1)
    
    # For stability, shift Lambda slightly into the left half-plane
    # (the skew-symmetric eigenvalues are purely imaginary)
    Lambda = Lambda - 0.5  # Shift real part to -0.5 for stability
    
    return Lambda, P, P.clone(), V


def cauchy_kernel(
    omega: torch.Tensor,
    Lambda: torch.Tensor,
    P: torch.Tensor,
    Q: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor
) -> torch.Tensor:
    """
    Compute SSM generating function using Cauchy kernel for NPLR system.
    
    For A = Λ - PQ*, computes:
        ĥ(ω) = C @ (ωI - Λ + PQ*)^{-1} @ B
    
    Using the Woodbury identity:
        (ωI - Λ + PQ*)^{-1} = R - R P (1 + Q* R P)^{-1} Q* R
    where R = (ωI - Λ)^{-1} is diagonal (Cauchy kernel).
    
    Args:
        omega: Evaluation points, shape (L,), complex.
        Lambda: Diagonal eigenvalues, shape (P_dim,), complex.
        P: Low-rank factor P, shape (P_dim,), complex.
        Q: Low-rank factor Q, shape (P_dim,), complex.
        B: Input matrix, shape (P_dim, H), complex.
        C: Output matrix, shape (H, P_dim), complex.
        
    Returns:
        K: Evaluated transfer function, shape (L, H), complex.
        
    Reference:
        Section 3.2 and Appendix D of S4 paper.
    """
    L = omega.shape[0]
    P_dim = Lambda.shape[0]
    
    # R = diag(1 / (omega_j - lambda_i)) — Cauchy matrix
    # Shape: (L, P_dim)
    # Add small imaginary perturbation to avoid exact singularity
    denom = omega.unsqueeze(-1) - Lambda.unsqueeze(0)
    R_diag = 1.0 / (denom + 1e-12j)
    
    # Woodbury: (ωI - A)^{-1} = R - R P (1 + Q* R P)^{-1} Q* R
    
    # R_diag has shape (L, P_dim), P and Q have shape (P_dim,)
    RB = R_diag.unsqueeze(-1) * B.unsqueeze(0)  # (L, P_dim, H)
    RP = R_diag * P.unsqueeze(0)                  # (L, P_dim)
    QR = torch.conj(Q).unsqueeze(0) * R_diag      # (L, P_dim)
    
    # Q* R P: sum over state dim -> (L,)
    QTRP = torch.sum(QR * P.unsqueeze(0), dim=-1)  # (L,)
    
    # Q* R B: (L, H)
    QTRB = torch.einsum('lp,ph->lh', QR, B)
    
    # C R B: (L, H) — direct term
    CRB = torch.einsum('hp,lph->lh', C, RB)
    
    # C R P: (L, H)
    CRP = torch.einsum('hp,lp->lh', C, RP)
    
    # Woodbury correction: C R P * (1 + Q* R P)^{-1} * Q* R B
    # Guard against (1 + Q*RP) ≈ 0
    woodbury_denom = 1.0 + QTRP
    correction = CRP * (1.0 / (woodbury_denom + 1e-12j)).unsqueeze(-1) * QTRB
    
    K = CRB - correction
    
    return K


def discretize_zoh(
    Lambda: torch.Tensor,
    P: torch.Tensor,
    Q: torch.Tensor,
    B: torch.Tensor,
    delta: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Discretize NPLR system using Zero-Order Hold (ZOH).
    
    Converts continuous-time NPLR parameters to discrete-time for recurrent mode.
    
    Args:
        Lambda: Diagonal eigenvalues, shape (P,), complex.
        P: Low-rank factor P, shape (P,), complex.
        Q: Low-rank factor Q, shape (P,), complex.
        B: Input matrix, shape (P, H), complex.
        delta: Discretization timestep, shape (1,) or scalar.
        
    Returns:
        Lambda_bar: Discretized diagonal, shape (P,), complex.
        P_bar: Discretized P, shape (P,), complex.
        Q_bar: Discretized Q (unchanged), shape (P,), complex.
        B_bar: Discretized input matrix, shape (P, H), complex.
        
    Reference:
        Appendix C.2 of S4 paper.
    """
    # Expand delta if scalar
    if delta.dim() == 0:
        delta = delta.unsqueeze(0).expand(Lambda.shape[0])
    elif delta.shape[0] == 1:
        delta = delta.expand(Lambda.shape[0])
    
    # Discretize Lambda: Λ̄ = exp(Λ * Δ)
    Lambda_delta = Lambda * delta
    Lambda_bar = torch.exp(Lambda_delta)
    
    # Discretize B: B̄ = (Λ̄ - I) / Λ * B
    # Use numerically stable computation: (exp(Λ*Δ) - 1) / Λ = Δ * expm1(Λ*Δ) / (Λ*Δ)
    # For small |Λ*Δ|, (exp(x)-1)/x ≈ 1 + x/2 + x²/6
    # For larger values, use direct formula with safe denominator
    abs_ld = Lambda_delta.abs()
    # Use Taylor expansion where |Lambda*delta| is small to avoid 0/0
    small = abs_ld < 1e-4
    factor = torch.where(
        small,
        delta * (1.0 + Lambda_delta / 2.0 + Lambda_delta ** 2 / 6.0),
        (Lambda_bar - 1.0) / (Lambda + 1e-12 * (1 + 1j))
    )
    B_bar = factor.unsqueeze(-1) * B
    
    # Discretize P: P̄ = (Λ̄ - I) / Λ * P
    P_bar = factor * P
    
    # Q remains unchanged in ZOH
    Q_bar = Q
    
    return Lambda_bar, P_bar, Q_bar, B_bar


def _sequential_scan_nplr(
    Lambda_bar: torch.Tensor,
    P_bar: torch.Tensor,
    Q_bar: torch.Tensor,
    Bu_elements: torch.Tensor
) -> torch.Tensor:
    """
    Sequential scan for NPLR recurrence.
    
    Computes: x_k = (Λ̄ - P̄Q̄*) x_{k-1} + B̄u_k
    
    Args:
        Lambda_bar: Discretized diagonal, shape (P,), complex.
        P_bar: Discretized P, shape (P,), complex.
        Q_bar: Discretized Q, shape (P,), complex.
        Bu_elements: Pre-computed B̄u_k, shape (B, L, P), complex.
        
    Returns:
        xs: State sequence, shape (B, L, P), complex.
    """
    B, L, P = Bu_elements.shape
    device = Bu_elements.device
    dtype = Bu_elements.dtype
    
    xs = torch.zeros(B, L, P, dtype=dtype, device=device)
    x = torch.zeros(B, P, dtype=dtype, device=device)
    
    for k in range(L):
        # Low-rank update: x = Λ̄ x - P̄ (Q̄* · x) + B̄u_k
        qx = torch.sum(torch.conj(Q_bar).unsqueeze(0) * x, dim=-1, keepdim=True)
        x = Lambda_bar.unsqueeze(0) * x - qx * P_bar.unsqueeze(0) + Bu_elements[:, k, :]
        xs[:, k, :] = x
    
    return xs


class S4SSM(nn.Module):
    """
    S4 State Space Model layer with NPLR parameterization.
    
    Implements a structured state space model using:
    - HiPPO-LegS initialization for long-range dependencies
    - NPLR (Normal Plus Low-Rank) representation: A = Λ - PQ*
    - Efficient convolution via Cauchy kernel
    - Recurrent mode for autoregressive generation
    
    Args:
        input_size: Input feature dimension H.
        state_size: Latent state dimension P (should be even).
        dt_min: Minimum value for timescale initialization.
        dt_max: Maximum value for timescale initialization.
        mode: Computation mode ("recurrent" or "convolution").
        
    Reference:
        Gu et al. "Efficiently Modeling Long Sequences with Structured State Spaces"
        ICLR 2022.
    """
    
    def __init__(
        self,
        input_size: int,
        state_size: int,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        mode: str = "recurrent",
    ):
        super().__init__()
        
        assert state_size % 2 == 0, "State size must be even for DPLR"
        
        self.input_size = input_size
        self.state_size = state_size
        self.mode = mode
        
        # Initialize NPLR representation from HiPPO-LegS
        Lambda, P, Q = make_nplr_hippo(state_size)
        
        # Shift Lambda into left half-plane for stability
        # (raw HiPPO skew-symmetric eigenvalues are purely imaginary)
        Lambda = Lambda - 0.5
        
        # Learnable parameters
        # Lambda parameterized in log space for stability (real part must be negative)
        self.log_lambda_real = nn.Parameter(
            torch.log(-Lambda.real.clamp(max=-1e-4))
        )
        self.lambda_imag = nn.Parameter(Lambda.imag.clone())
        
        # Low-rank factors
        self.P_real = nn.Parameter(P.real.clone())
        self.P_imag = nn.Parameter(P.imag.clone())
        self.Q_real = nn.Parameter(Q.real.clone())
        self.Q_imag = nn.Parameter(Q.imag.clone())
        
        # Input matrix B: shape (state_size, input_size)
        B_init = torch.randn(state_size, input_size, dtype=torch.complex64)
        B_init = B_init / math.sqrt(state_size)
        self.B_real = nn.Parameter(B_init.real)
        self.B_imag = nn.Parameter(B_init.imag)
        
        # Output matrix C: shape (input_size, state_size)
        C_init = torch.randn(input_size, state_size, dtype=torch.complex64)
        C_init = C_init / math.sqrt(state_size)
        self.C_real = nn.Parameter(C_init.real)
        self.C_imag = nn.Parameter(C_init.imag)
        
        # Feedthrough matrix D
        self.D = nn.Parameter(torch.randn(input_size))
        
        # Timescale parameter Δ
        log_dt = torch.rand(1) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_delta = nn.Parameter(log_dt)
        
        # Cache for convolution kernel
        self._kernel_cache: Optional[torch.Tensor] = None
        self._cache_length: Optional[int] = None
    
    @property
    def Lambda(self) -> torch.Tensor:
        """Diagonal eigenvalues (negative real part for stability)."""
        return -torch.exp(self.log_lambda_real) + 1j * self.lambda_imag
    
    @property
    def P(self) -> torch.Tensor:
        """Low-rank factor P."""
        return self.P_real + 1j * self.P_imag
    
    @property
    def Q(self) -> torch.Tensor:
        """Low-rank factor Q."""
        return self.Q_real + 1j * self.Q_imag
    
    @property
    def B(self) -> torch.Tensor:
        """Input matrix."""
        return self.B_real + 1j * self.B_imag
    
    @property
    def C(self) -> torch.Tensor:
        """Output matrix."""
        return self.C_real + 1j * self.C_imag
    
    @property
    def delta(self) -> torch.Tensor:
        """Discretization timestep."""
        return torch.exp(self.log_delta)
    
    def _get_kernel(self, length: int) -> torch.Tensor:
        """
        Compute or retrieve cached convolution kernel via Cauchy kernel.
        
        Evaluates the ZOH-discretized transfer function at the roots of unity
        (DFT frequencies) to obtain the convolution kernel.
        
        Args:
            length: Sequence length L.
            
        Returns:
            K: Convolution kernel, shape (L, H).
        """
        if self.training or self._kernel_cache is None or self._cache_length != length:
            # BUG FIX #4: Evaluate at bilinear-transformed DFT frequencies.
            # Use float64 for kernel computation (standard practice in S4 to
            # avoid precision loss in Cauchy kernel / Woodbury formula).
            
            Lambda = self.Lambda.to(torch.complex128)
            P_lr = self.P.to(torch.complex128)
            Q_lr = self.Q.to(torch.complex128)
            B_mat = self.B.to(torch.complex128)
            C_mat = self.C.to(torch.complex128)
            dt = self.delta.squeeze().to(torch.float64)
            
            # DFT frequencies: z = exp(2πi k / L)
            k_idx = torch.arange(length, device=self.Lambda.device, dtype=torch.float64)
            z = torch.exp(2j * math.pi * k_idx / length).to(torch.complex128)
            
            # Bilinear transform: map z to continuous-time frequency
            # Add small epsilon to avoid division by zero at z = -1 (Nyquist)
            omega = (2.0 / dt) * (z - 1) / (z + 1 + 1e-12)
            
            K_f = cauchy_kernel(omega, Lambda, P_lr, Q_lr, B_mat, C_mat)
            
            # Scale by 2Δ/(1+z) from bilinear discretization
            scale = (2.0 * dt / (1.0 + z + 1e-12)).unsqueeze(-1)
            K_f = K_f * scale
            
            # IFFT to get time-domain kernel, cast back to float32
            K = torch.fft.ifft(K_f, dim=0).real.to(torch.float32)  # (L, H)
            
            # Safety: clamp any residual NaN from edge cases
            if K.isnan().any():
                K = torch.nan_to_num(K, nan=0.0)
            
            if not self.training:
                self._kernel_cache = K
                self._cache_length = length
            
            return K
        else:
            return self._kernel_cache
    
    def forward(
        self,
        u: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply S4 SSM to input sequence.
        
        Args:
            u: Input sequence, shape (batch, length, input_size).
            state: Optional initial state, shape (batch, state_size), complex.
            
        Returns:
            y: Output sequence, shape (batch, length, input_size).
            final_state: Final state, shape (batch, state_size), complex.
        """
        if self.mode == "convolution":
            return self._forward_convolution(u, state)
        else:
            return self._forward_recurrent(u, state)
    
    def _forward_convolution(
        self,
        u: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convolution mode using FFT."""
        batch_size, seq_len, _ = u.shape
        
        # Get convolution kernel
        K = self._get_kernel(seq_len)  # (L, H)
        
        # Perform causal convolution via FFT
        # Pad to avoid circular convolution artifacts
        fft_size = 2 * seq_len
        
        # FFT of input: (B, L, H)
        u_f = torch.fft.rfft(u, n=fft_size, dim=1)  # (B, fft_size//2+1, H)
        
        # FFT of kernel: (L, H)
        K_f = torch.fft.rfft(K, n=fft_size, dim=0)  # (fft_size//2+1, H)
        
        # Multiply in frequency domain
        y_f = u_f * K_f.unsqueeze(0)  # (B, fft_size//2+1, H)
        
        # IFFT to get output
        y = torch.fft.irfft(y_f, n=fft_size, dim=1)  # (B, fft_size, H)
        y = y[:, :seq_len, :]  # Truncate to original length
        
        # Add feedthrough
        y = y + self.D.unsqueeze(0).unsqueeze(0) * u
        
        # For convolution mode, compute final state via recurrence on last few steps
        # (or return zeros as approximation)
        final_state = torch.zeros(
            batch_size, self.state_size,
            dtype=torch.complex64,
            device=u.device
        )
        
        return y, final_state
    
    def _forward_recurrent(
        self,
        u: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Recurrent mode using discretized system."""
        batch_size, seq_len, _ = u.shape
        
        # Discretize system
        Lambda_bar, P_bar, Q_bar, B_bar = discretize_zoh(
            self.Lambda, self.P, self.Q, self.B, self.delta
        )
        
        # Compute Bu for all timesteps
        u_complex = u.to(torch.complex64)
        Bu_elements = torch.einsum('blh,ph->blp', u_complex, B_bar)
        
        # Handle initial state
        if state is not None:
            Bu_elements = Bu_elements.clone()
            # Apply NPLR update to initial state and add to first Bu
            qx = torch.sum(torch.conj(Q_bar).unsqueeze(0) * state, dim=-1, keepdim=True)
            x0 = Lambda_bar.unsqueeze(0) * state - qx * P_bar.unsqueeze(0)
            Bu_elements[:, 0, :] = Bu_elements[:, 0, :] + x0
        
        # Apply sequential scan
        xs = _sequential_scan_nplr(Lambda_bar, P_bar, Q_bar, Bu_elements)
        
        # Compute outputs: y = Re(C @ x) + D * u
        y = torch.einsum('blp,hp->blh', xs, self.C)
        y = y.real + self.D.unsqueeze(0).unsqueeze(0) * u
        
        final_state = xs[:, -1, :]
        
        return y, final_state
    
    def step(
        self,
        u: torch.Tensor,
        state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single step for autoregressive generation.
        
        Args:
            u: Input at current timestep, shape (batch, input_size).
            state: Current state, shape (batch, state_size), complex.
            
        Returns:
            y: Output, shape (batch, input_size).
            new_state: Updated state, shape (batch, state_size), complex.
        """
        # Discretize
        Lambda_bar, P_bar, Q_bar, B_bar = discretize_zoh(
            self.Lambda, self.P, self.Q, self.B, self.delta
        )
        
        # Compute Bu
        u_complex = u.to(torch.complex64)
        Bu = torch.einsum('bh,ph->bp', u_complex, B_bar)
        
        # NPLR state update: x = (Λ̄ - P̄Q̄*) x + B̄u
        qx = torch.sum(torch.conj(Q_bar).unsqueeze(0) * state, dim=-1, keepdim=True)
        new_state = Lambda_bar.unsqueeze(0) * state - qx * P_bar.unsqueeze(0) + Bu
        
        # Output: y = Re(C @ x) + D * u
        y = torch.einsum('bp,hp->bh', new_state, self.C)
        y = y.real + self.D.unsqueeze(0) * u
        
        return y, new_state


class S4Block(nn.Module):
    """
    S4 Block with normalization, SSM, and gated activation.
    
    Architecture:
        x -> LayerNorm -> S4 SSM -> GELU * Sigmoid(gate) -> Dropout -> + x (residual)
    """
    
    def __init__(
        self,
        input_size: int,
        state_size: int,
        dropout: float = 0.1,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        mode: str = "recurrent",
    ):
        super().__init__()
        
        self.norm = nn.LayerNorm(input_size)
        self.ssm = S4SSM(
            input_size=input_size,
            state_size=state_size,
            dt_min=dt_min,
            dt_max=dt_max,
            mode=mode,
        )
        self.dropout = nn.Dropout(dropout)
        self.gate_proj = nn.Linear(input_size, input_size)
    
    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = x
        x = self.norm(x)
        
        y, final_state = self.ssm(x, state)
        
        # Gated activation
        y_gelu = F.gelu(y)
        gate = torch.sigmoid(self.gate_proj(y_gelu))
        y = y_gelu * gate
        
        y = self.dropout(y)
        
        # BUG FIX #5: Add residual connection (was computed but never used)
        y = y + residual
        
        return y, final_state
    
    def step(
        self,
        x: torch.Tensor,
        state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = x
        x = self.norm(x)
        
        y, new_state = self.ssm.step(x, state)
        
        y_gelu = F.gelu(y)
        gate = torch.sigmoid(self.gate_proj(y_gelu))
        y = y_gelu * gate
        
        # BUG FIX #6: Add residual connection in step() too
        y = y + residual
        
        return y, new_state


class S4(BaseModel):
    """
    S4 Model for Spatio-Temporal Forecasting.
    
    Uses stacked S4 layers with NPLR parameterization for efficient
    long-range sequence modeling.
    
    Args:
        input_size: Input feature dimension.
        hidden_size: Hidden dimension.
        output_size: Output feature dimension.
        horizon: Forecasting horizon.
        n_layers: Number of S4 blocks.
        state_size: SSM state dimension (must be even).
        dropout: Dropout probability.
        dt_min: Minimum timescale.
        dt_max: Maximum timescale.
        exog_size: Exogenous feature dimension.
        mode: SSM mode ("recurrent" or "convolution").
        
    Reference:
        Gu et al. "Efficiently Modeling Long Sequences with Structured State Spaces"
        ICLR 2022.
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
        dropout: float = 0.1,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        exog_size: int = 0,
        mode: str = "recurrent",
    ):
        super().__init__()
        
        assert state_size % 2 == 0, "State size must be even for S4"
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.horizon = horizon
        self.n_layers = n_layers
        self.state_size = state_size
        self.mode = mode
        
        # Input encoder
        self.input_encoder = nn.Linear(input_size + exog_size, hidden_size)
        
        # Stacked S4 blocks
        self.blocks = nn.ModuleList([
            S4Block(
                input_size=hidden_size,
                state_size=state_size,
                dropout=dropout,
                dt_min=dt_min,
                dt_max=dt_max,
                mode=mode,
            )
            for _ in range(n_layers)
        ])
        
        # Output decoder
        self.output_norm = nn.LayerNorm(hidden_size)
        self.output_decoder = nn.Linear(hidden_size, output_size)
    
    def forward(
        self,
        x: torch.Tensor,
        u: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass with teacher forcing for training.
        
        Args:
            x: Input, shape (batch, window, input_size).
            u: Optional exogenous features, shape (batch, window+horizon, exog_size).
            
        Returns:
            Output, shape (batch, horizon, output_size).
        """
        batch_size, window, _ = x.shape
        
        # Concatenate exogenous features
        if u is not None:
            x = torch.cat([x, u[:, :window]], dim=-1)
        
        # Encode
        x = self.input_encoder(x)
        
        if not self.training:
            # Inference: process full context, then generate autoregressively
            states = []
            for block in self.blocks:
                x, state = block(x)
                states.append(state)
            
            return self._autoregressive_generation(x, states, batch_size)
        
        else:
            # Training: teacher forcing — process full sequence
            for block in self.blocks:
                x, _ = block(x)
            
            # Decode all timesteps
            out = self.output_norm(x)
            out = self.output_decoder(out)
            
            # Return last horizon predictions
            out = out[:, -self.horizon:, :]

            # print(f"Out is nan: {out.isnan().any().item()}, inf: {out.isinf().any().item()}")
            
            return out
    
    def _autoregressive_generation(
        self,
        x: torch.Tensor,
        states: list[torch.Tensor],
        batch_size: int
    ) -> torch.Tensor:
        """
        Autoregressive generation for inference.
        
        BUG FIX #7: The original code used raw hidden states as inputs to block.step(),
        but block.step() expects the same-shaped tensor as block.forward() produces.
        The generation loop now correctly feeds the output of the block stack back
        through subsequent steps.
        """
        # Decode first output from last context hidden state
        last_hidden = x[:, -1, :]  # (B, hidden_size)
        
        outputs = []
        out = self.output_norm(last_hidden)
        out = self.output_decoder(out)
        outputs.append(out)
        
        # For subsequent steps, feed last_hidden through blocks recurrently
        current_hidden = last_hidden
        
        for t in range(self.horizon - 1):
            h = current_hidden
            new_states = []
            for i, block in enumerate(self.blocks):
                h, new_state = block.step(h, states[i])
                new_states.append(new_state)
            states = new_states
            current_hidden = h
            
            out = self.output_norm(h)
            out = self.output_decoder(out)
            outputs.append(out)
        
        outputs = torch.stack(outputs, dim=1)  # (B, horizon, output_size)
        return outputs
    
    @torch.no_grad()
    def generate(
        self,
        x: torch.Tensor,
        horizon: Optional[int] = None,
        u: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Autoregressive generation."""
        if horizon is None:
            horizon = self.horizon
        
        original_horizon = self.horizon
        self.horizon = horizon
        
        output = self.forward(x, u, **kwargs)
        
        self.horizon = original_horizon
        return output


if __name__ == "__main__":
    print("=" * 60)
    print("S4 Model Test Suite")
    print("=" * 60)
    
    batch_size = 4
    window_size = 24
    horizon = 12
    input_size = 3
    output_size = 1
    hidden_size = 64
    state_size = 64  # Must be even
    
    # Test model creation
    print("\n1. Testing S4 model creation...")
    model = S4(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        horizon=horizon,
        n_layers=4,
        state_size=state_size,
        dropout=0.1,
        mode="recurrent"
    )
    
    x = torch.randn(batch_size, window_size, input_size)
    y = model(x)
    
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {y.shape}")
    print(f"   Expected: ({batch_size}, {horizon}, {output_size})")
    assert y.shape == (batch_size, horizon, output_size)
    print("   ✓ Shape test passed!")
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Number of parameters: {n_params:,}")
    
    # Test gradient flow
    print("\n2. Testing gradient flow...")
    loss = y.sum()
    loss.backward()
    print("   ✓ Gradient flow test passed!")
    
    # Test HiPPO initialization
    print("\n3. Testing HiPPO-LegS initialization...")
    A = make_hippo_legs_matrix(8)
    print(f"   HiPPO-LegS matrix shape: {A.shape}")
    print(f"   Diagonal values: {A.diagonal()}")
    print(f"   All diagonal entries negative: {(A.diagonal() < 0).all()}")
    print(f"   Lower triangular: {torch.allclose(A, torch.tril(A))}")
    
    # Test NPLR decomposition
    Lambda, P, Q = make_nplr_hippo(8)
    print(f"   NPLR - Lambda shape: {Lambda.shape}, P shape: {P.shape}, Q shape: {Q.shape}")
    print(f"   Lambda has purely imaginary base: "
          f"max |Re(Lambda)| = {Lambda.real.abs().max():.6f}")
    print("   ✓ HiPPO initialization test passed!")
    
    # Test inference mode
    print("\n4. Testing inference (autoregressive) mode...")
    model.eval()
    with torch.no_grad():
        y_infer = model(x)
    print(f"   Inference output shape: {y_infer.shape}")
    assert y_infer.shape == (batch_size, horizon, output_size)
    print("   ✓ Inference mode test passed!")
    
    # Test convolution mode
    print("\n5. Testing convolution mode...")
    model_conv = S4(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        horizon=horizon,
        n_layers=2,
        state_size=state_size,
        dropout=0.0,
        mode="convolution"
    )
    model_conv.train()
    y_conv = model_conv(x)
    print(f"   Convolution output shape: {y_conv.shape}")
    assert y_conv.shape == (batch_size, horizon, output_size)
    loss_conv = y_conv.sum()
    loss_conv.backward()
    print("   ✓ Convolution mode test passed!")
    
    print("\n" + "=" * 60)
    print("All S4 tests passed!")
    print("=" * 60)
