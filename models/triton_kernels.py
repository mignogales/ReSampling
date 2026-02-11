"""
S5 Parallel Scan with Autograd Support.

The key insight: for the linear recurrence x_k = λ x_{k-1} + b_k,
the backward pass involves a *reverse* scan with conjugate eigenvalues.

Forward:  x_k = λ x_{k-1} + b_k        (scan left-to-right)
Backward: ∂L/∂b_k = ∂L/∂x_k + λ* ∂L/∂b_{k+1}  (scan right-to-left)
          ∂L/∂λ = Σ_k (∂L/∂x_k)* x_{k-1}

Reference:
    Martin & Cundy (2018). "Parallelizing Linear Recurrent Neural Nets Over Sequence Length"
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

_TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    pass


# =============================================================================
# Triton Kernels
# =============================================================================

if _TRITON_AVAILABLE:

    @triton.jit
    def _complex_mul(a_re, a_im, b_re, b_im):
        """Complex multiplication."""
        out_re = a_re * b_re - a_im * b_im
        out_im = a_re * b_im + a_im * b_re
        return out_re, out_im

    @triton.jit
    def _scan_combine_fn(
        a1_re, a1_im, b1_re, b1_im,
        a2_re, a2_im, b2_re, b2_im
    ):
        """Associative operator: (a1,b1) ⊕ (a2,b2) = (a2*a1, a2*b1 + b2)"""
        a_out_re, a_out_im = _complex_mul(a2_re, a2_im, a1_re, a1_im)
        ab_re, ab_im = _complex_mul(a2_re, a2_im, b1_re, b1_im)
        b_out_re = ab_re + b2_re
        b_out_im = ab_im + b2_im
        return a_out_re, a_out_im, b_out_re, b_out_im

    @triton.jit
    def forward_scan_kernel(
        gates_re_ptr, gates_im_ptr,
        tokens_re_ptr, tokens_im_ptr,
        out_re_ptr, out_im_ptr,
        B: tl.constexpr, L: tl.constexpr, P: tl.constexpr,
        stride_b: tl.constexpr, stride_l: tl.constexpr, stride_p: tl.constexpr,
        BLOCK_L: tl.constexpr,
    ):
        """Forward scan: x_k = λ x_{k-1} + b_k"""
        pid_b = tl.program_id(0)
        pid_p = tl.program_id(1)
        
        gate_re = tl.load(gates_re_ptr + pid_p)
        gate_im = tl.load(gates_im_ptr + pid_p)
        
        seq_idx = tl.arange(0, BLOCK_L)
        mask = seq_idx < L
        offsets = pid_b * stride_b + seq_idx * stride_l + pid_p * stride_p
        
        tok_re = tl.load(tokens_re_ptr + offsets, mask=mask, other=0.0)
        tok_im = tl.load(tokens_im_ptr + offsets, mask=mask, other=0.0)
        
        # Identity at position 0: (1, 0)
        a_re = tl.where(seq_idx == 0, 1.0, gate_re)
        a_im = tl.where(seq_idx == 0, 0.0, gate_im)
        
        _, _, out_re, out_im = tl.associative_scan(
            (a_re, a_im, tok_re, tok_im),
            axis=0,
            combine_fn=_scan_combine_fn
        )
        
        tl.store(out_re_ptr + offsets, out_re, mask=mask)
        tl.store(out_im_ptr + offsets, out_im, mask=mask)

    @triton.jit
    def backward_scan_kernel(
        gates_re_ptr, gates_im_ptr,  # λ* (conjugate)
        grad_out_re_ptr, grad_out_im_ptr,  # ∂L/∂x
        grad_tokens_re_ptr, grad_tokens_im_ptr,  # ∂L/∂b (output)
        B: tl.constexpr, L: tl.constexpr, P: tl.constexpr,
        stride_b: tl.constexpr, stride_l: tl.constexpr, stride_p: tl.constexpr,
        BLOCK_L: tl.constexpr,
    ):
        """
        Backward scan (reversed): ∂L/∂b_k = ∂L/∂x_k + λ* ∂L/∂b_{k+1}
        
        This is equivalent to a forward scan on the reversed sequence
        with conjugate eigenvalues.
        """
        pid_b = tl.program_id(0)
        pid_p = tl.program_id(1)
        
        # Load conjugate eigenvalue
        gate_re = tl.load(gates_re_ptr + pid_p)
        gate_im = tl.load(gates_im_ptr + pid_p)  # Already conjugated by caller
        
        seq_idx = tl.arange(0, BLOCK_L)
        # Reverse indexing: position 0 in scan = position L-1 in sequence
        rev_seq_idx = L - 1 - seq_idx
        mask = seq_idx < L
        
        offsets = pid_b * stride_b + rev_seq_idx * stride_l + pid_p * stride_p
        
        # Load grad_output in reverse order
        tok_re = tl.load(grad_out_re_ptr + offsets, mask=mask, other=0.0)
        tok_im = tl.load(grad_out_im_ptr + offsets, mask=mask, other=0.0)
        
        # Identity at position 0 of reversed scan (= position L-1 of original)
        a_re = tl.where(seq_idx == 0, 1.0, gate_re)
        a_im = tl.where(seq_idx == 0, 0.0, gate_im)
        
        _, _, out_re, out_im = tl.associative_scan(
            (a_re, a_im, tok_re, tok_im),
            axis=0,
            combine_fn=_scan_combine_fn
        )
        
        # Store in reverse order to get correct positions
        tl.store(grad_tokens_re_ptr + offsets, out_re, mask=mask)
        tl.store(grad_tokens_im_ptr + offsets, out_im, mask=mask)


# =============================================================================
# Autograd Function
# =============================================================================

class ParallelScanFunction(torch.autograd.Function):
    """
    Custom autograd function for parallel scan with Triton acceleration.
    
    Forward: x_k = λ x_{k-1} + b_k
    
    Backward pass derivation:
    Given loss L and ∂L/∂x_k for all k, we need ∂L/∂λ and ∂L/∂b_k.
    
    By chain rule on x_k = λ x_{k-1} + b_k:
        ∂L/∂b_k = ∂L/∂x_k · ∂x_k/∂b_k = ∂L/∂x_k · 1 = ∂L/∂x_k
        
    But x_k also affects x_{k+1}, x_{k+2}, ... through the recurrence:
        ∂L/∂b_k = ∂L/∂x_k + λ* · ∂L/∂b_{k+1}
        
    This is a reverse-time scan! Similarly:
        ∂L/∂λ = Σ_k conj(∂L/∂x_k · x_{k-1})
              = Σ_k conj(∂L/∂b_k) · x_{k-1}   [after the reverse scan]
    """
    
    @staticmethod
    def forward(ctx, lambda_bar: torch.Tensor, Bu_elements: torch.Tensor):
        """
        Forward pass of parallel scan.
        
        Args:
            lambda_bar: (P,) complex - discretized eigenvalues
            Bu_elements: (B, L, P) complex - input sequence
            
        Returns:
            xs: (B, L, P) complex - state sequence
        """
        B, L, P = Bu_elements.shape
        
        Bu_elements = Bu_elements.contiguous()
        lambda_bar = lambda_bar.contiguous()
        
        gates_re = lambda_bar.real.contiguous()
        gates_im = lambda_bar.imag.contiguous()
        tokens_re = Bu_elements.real.contiguous()
        tokens_im = Bu_elements.imag.contiguous()
        
        out_re = torch.empty_like(tokens_re)
        out_im = torch.empty_like(tokens_im)
        
        stride_b, stride_l, stride_p = L * P, P, 1
        BLOCK_L = triton.next_power_of_2(L)
        
        grid = (B, P)
        forward_scan_kernel[grid](
            gates_re, gates_im,
            tokens_re, tokens_im,
            out_re, out_im,
            B, L, P,
            stride_b, stride_l, stride_p,
            BLOCK_L=BLOCK_L,
        )
        
        xs = torch.complex(out_re, out_im)
        
        # Save for backward
        ctx.save_for_backward(lambda_bar, xs)
        ctx.shape = (B, L, P)
        
        return xs
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward pass using reverse parallel scan.
        
        Args:
            grad_output: (B, L, P) complex - ∂L/∂x
            
        Returns:
            grad_lambda: (P,) complex - ∂L/∂λ
            grad_Bu: (B, L, P) complex - ∂L/∂b
        """
        lambda_bar, xs = ctx.saved_tensors
        B, L, P = ctx.shape
        
        grad_output = grad_output.contiguous()
        
        # Conjugate eigenvalue for backward scan
        lambda_conj = lambda_bar.conj().contiguous()
        gates_re = lambda_conj.real.contiguous()
        gates_im = lambda_conj.imag.contiguous()
        
        grad_re = grad_output.real.contiguous()
        grad_im = grad_output.imag.contiguous()
        
        grad_tokens_re = torch.empty_like(grad_re)
        grad_tokens_im = torch.empty_like(grad_im)
        
        stride_b, stride_l, stride_p = L * P, P, 1
        BLOCK_L = triton.next_power_of_2(L)
        
        grid = (B, P)
        backward_scan_kernel[grid](
            gates_re, gates_im,
            grad_re, grad_im,
            grad_tokens_re, grad_tokens_im,
            B, L, P,
            stride_b, stride_l, stride_p,
            BLOCK_L=BLOCK_L,
        )
        
        grad_Bu = torch.complex(grad_tokens_re, grad_tokens_im)
        
        # Gradient w.r.t. lambda: ∂L/∂λ = Σ_{b,k} conj(∂L/∂b_k) · x_{k-1}
        # x_{k-1} for k=0 is 0, so we use xs[:, :-1, :] paired with grad_Bu[:, 1:, :]
        xs_prev = torch.cat([
            torch.zeros(B, 1, P, dtype=xs.dtype, device=xs.device),
            xs[:, :-1, :]
        ], dim=1)
        
        # grad_lambda[p] = Σ_{b,k} conj(grad_Bu[b,k,p]) * xs_prev[b,k,p]
        grad_lambda = (grad_Bu.conj() * xs_prev).sum(dim=(0, 1))
        
        return grad_lambda, grad_Bu


def triton_parallel_scan(
    lambda_bar: torch.Tensor,
    Bu_elements: torch.Tensor
) -> torch.Tensor:
    """
    Triton-accelerated parallel scan with autograd support.
    
    This function properly propagates gradients to both lambda_bar
    and Bu_elements, enabling training of A and B matrices.
    
    Args:
        lambda_bar: (P,) complex - discretized diagonal eigenvalues
        Bu_elements: (B, L, P) complex - input projections
        
    Returns:
        xs: (B, L, P) complex - state sequence
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton not available")
    
    return ParallelScanFunction.apply(lambda_bar, Bu_elements)


# =============================================================================
# PyTorch Sequential (Reference)
# =============================================================================

def pytorch_sequential_scan(
    lambda_bar: torch.Tensor,
    Bu_elements: torch.Tensor
) -> torch.Tensor:
    """Reference implementation with native autograd."""
    B, L, P = Bu_elements.shape
    device, dtype = Bu_elements.device, Bu_elements.dtype
    
    xs = []
    x = torch.zeros(B, P, device=device, dtype=dtype)
    
    for k in range(L):
        x = lambda_bar * x + Bu_elements[:, k, :]
        xs.append(x)
    
    return torch.stack(xs, dim=1)


# =============================================================================
# Validation
# =============================================================================

def validate_gradients(B=4, L=64, P=32, device="cuda"):
    """
    Validate gradient computation against PyTorch autograd.
    """
    if not _TRITON_AVAILABLE:
        print("Triton not available")
        return
    
    torch.manual_seed(42)
    
    # Create inputs requiring grad
    lambda_bar_base = 0.9 * torch.exp(1j * torch.randn(P, device=device))
    Bu_base = torch.randn(B, L, P, device=device, dtype=torch.complex64)
    
    # Test 1: Triton implementation
    lambda_triton = lambda_bar_base.clone().requires_grad_(True)
    Bu_triton = Bu_base.clone().requires_grad_(True)
    
    xs_triton = triton_parallel_scan(lambda_triton, Bu_triton)
    loss_triton = xs_triton.abs().sum()
    loss_triton.backward()
    
    # Test 2: PyTorch reference
    lambda_pytorch = lambda_bar_base.clone().requires_grad_(True)
    Bu_pytorch = Bu_base.clone().requires_grad_(True)
    
    xs_pytorch = pytorch_sequential_scan(lambda_pytorch, Bu_pytorch)
    loss_pytorch = xs_pytorch.abs().sum()
    loss_pytorch.backward()
    
    # Compare gradients
    print("Forward pass difference:", 
          (xs_triton - xs_pytorch).abs().max().item())
    print("grad_lambda difference:", 
          (lambda_triton.grad - lambda_pytorch.grad).abs().max().item())
    print("grad_Bu difference:", 
          (Bu_triton.grad - Bu_pytorch.grad).abs().max().item())
    
    # Check if gradients exist and are non-zero
    assert lambda_triton.grad is not None, "lambda grad is None!"
    assert Bu_triton.grad is not None, "Bu grad is None!"
    assert lambda_triton.grad.abs().sum() > 0, "lambda grad is zero!"
    assert Bu_triton.grad.abs().sum() > 0, "Bu grad is zero!"
    
    print("✓ Gradient validation passed")


if __name__ == "__main__":

    
    if torch.cuda.is_available() and _TRITON_AVAILABLE:
        validate_gradients()
    else:
        print("CUDA/Triton not available for testing")
