"""
PatchTST: A Time Series is Worth 64 Words (Nie et al., ICLR 2023)

Channel-independent Patch Time Series Transformer for long-term forecasting.
Paper: https://arxiv.org/abs/2211.14730

Key innovations:
1. Patching: Segments time series into subseries-level patches as input tokens
2. Channel-independence: Each channel shares the same Transformer backbone
3. Enables longer look-back windows with quadratic reduction in complexity
"""

import torch
import torch.nn as nn
import einops
import math
from tsl.nn.models.base_model import BaseModel


class RevIN(nn.Module):
    """
    Reversible Instance Normalization.
    
    Normalizes input with instance statistics and can denormalize output
    to mitigate distribution shift between train/test data.
    """
    
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x, mode: str = 'norm'):
        """
        Args:
            x: [B, T, F] or stored statistics for denorm
            mode: 'norm' for normalization, 'denorm' for denormalization
        """
        if mode == 'norm':
            self._mean = x.mean(dim=1, keepdim=True)
            self._std = x.std(dim=1, keepdim=True) + self.eps
            x = (x - self._mean) / self._std
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
            return x
        elif mode == 'denorm':
            if self.affine:
                x = (x - self.affine_bias) / self.affine_weight
            x = x * self._std + self._mean
            return x
        else:
            raise ValueError(f"Unknown mode: {mode}")


class PatchEmbedding(nn.Module):
    """
    Patch embedding layer: segments time series and projects to latent space.
    """
    
    def __init__(
        self,
        patch_len: int,
        stride: int,
        d_model: int,
        input_dim: int,
        dropout: float = 0.0
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        
        # Linear projection from patch to d_model
        self.projection = nn.Linear(patch_len * input_dim, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: [B, T, F] input sequence
            
        Returns:
            patches: [B, num_patches, d_model]
        """
        B, T, F = x.shape
        
        # Pad sequence if necessary to ensure we can create patches
        pad_len = (self.stride - (T - self.patch_len) % self.stride) % self.stride
        if pad_len > 0:
            # Pad by repeating the last value (as described in paper)
            padding = x[:, -1:, :].repeat(1, pad_len, 1)
            x = torch.cat([x, padding], dim=1)
        
        # Create patches using unfold
        # unfold(dimension, size, step) -> [B, num_patches, F, patch_len]
        patches = x.unfold(1, self.patch_len, self.stride)
        
        # Reshape to [B, num_patches, patch_len * F]
        patches = einops.rearrange(patches, 'b n f p -> b n (p f)')
        
        # Project to d_model
        patches = self.projection(patches)
        patches = self.dropout(patches)
        
        return patches


class TransformerEncoderBlock(nn.Module):
    """
    Transformer encoder block with BatchNorm (as used in original PatchTST).
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # BatchNorm (paper shows it outperforms LayerNorm for time series)
        self.norm1 = nn.BatchNorm1d(d_model)
        self.norm2 = nn.BatchNorm1d(d_model)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: [B, N, D] where N is num_patches, D is d_model
        Returns:
            x: [B, N, D]
        """
        # Self-attention with residual
        attn_out, _ = self.self_attn(x, x, x)
        x = x + self.dropout(attn_out)
        
        # BatchNorm expects [B, D, N], so transpose
        x = self.norm1(x.transpose(1, 2)).transpose(1, 2)
        
        # Feed-forward with residual
        ff_out = self.ff(x)
        x = x + ff_out
        
        # BatchNorm
        x = self.norm2(x.transpose(1, 2)).transpose(1, 2)
        
        return x


class PatchTST(BaseModel):
    """
    PatchTST: Channel-independent Patch Time Series Transformer.
    
    Args:
        input_size: Number of input features per node (F)
        output_size: Number of output features per node
        horizon: Prediction horizon (T_pred)
        window_size: Look-back window size (L)
        patch_len: Length of each patch (P), default 16
        stride: Stride between patches (S), default 8
        d_model: Model dimension (D), default 128
        n_heads: Number of attention heads (H), default 16
        n_layers: Number of transformer layers, default 3
        d_ff: Feed-forward dimension, default 256
        dropout: Dropout rate, default 0.2
        exog_size: Size of exogenous features, default 0
        use_revin: Whether to use reversible instance normalization, default True
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        horizon: int,
        window_size: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 16,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.2,
        exog_size: int = 0,
        use_revin: bool = True,
        **kwargs
    ):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.horizon = horizon
        self.window_size = window_size
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.use_revin = use_revin
        
        total_features = input_size + exog_size
        
        # Calculate number of patches
        # N = floor((L - P) / S) + 1, with potential padding
        pad_len = (stride - (window_size - patch_len) % stride) % stride
        padded_len = window_size + pad_len
        self.num_patches = (padded_len - patch_len) // stride + 1
        
        # Reversible Instance Normalization
        if use_revin:
            self.revin = RevIN(total_features)
        
        # Patch embedding
        self.patch_embedding = PatchEmbedding(
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            input_dim=total_features,
            dropout=dropout
        )
        
        # Learnable positional encoding
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])
        
        # Output head: flatten all patch representations and project to output
        self.head = nn.Linear(d_model * self.num_patches, horizon * output_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Forward pass with channel-independence.
        
        Args:
            x: [B, T, N, F] or [B, T, F] input tensor
               B: batch size
               T: time steps (window_size)
               N: number of nodes/channels
               F: features per node
               
        Returns:
            output: [B, horizon, N, output_size] or [B, horizon, output_size]
        """
        original_shape = x.shape
        
        # Handle 3D input (no spatial dimension)
        if len(original_shape) == 3:
            x = x.unsqueeze(2)  # [B, T, 1, F]
        
        B, T, N, F = x.shape
        
        # Channel-independence: reshape to process each channel independently
        # [B, T, N, F] -> [B*N, T, F]
        x = einops.rearrange(x, 'b t n f -> (b n) t f')
        
        # Instance normalization (RevIN)
        if self.use_revin:
            x = self.revin(x, mode='norm')
        
        # Patch embedding: [B*N, T, F] -> [B*N, num_patches, d_model]
        x = self.patch_embedding(x)
        
        # Add positional embedding
        x = x + self.pos_embedding
        x = self.dropout(x)
        
        # Transformer encoder
        for layer in self.encoder_layers:
            x = layer(x)
        
        # Flatten and project to output
        # [B*N, num_patches, d_model] -> [B*N, num_patches * d_model]
        x = einops.rearrange(x, 'bn np d -> bn (np d)')
        
        # Linear head: [B*N, num_patches * d_model] -> [B*N, horizon * output_size]
        x = self.head(x)
        
        # Reshape to [B*N, horizon, output_size]
        x = einops.rearrange(x, 'bn (h o) -> bn h o', h=self.horizon, o=self.output_size)
        
        # Denormalize output (RevIN)
        if self.use_revin:
            x = self.revin(x, mode='denorm')
        
        # Reshape back to include spatial dimension
        # [B*N, horizon, output_size] -> [B, horizon, N, output_size]
        x = einops.rearrange(x, '(b n) h o -> b h n o', b=B, n=N)
        
        # Handle 3D output
        if len(original_shape) == 3:
            x = x.squeeze(2)  # [B, horizon, output_size]
        
        return x


# Convenience function for smaller datasets (reduced capacity to prevent overfitting)
def PatchTSTSmall(
    input_size: int,
    output_size: int,
    horizon: int,
    window_size: int,
    exog_size: int = 0,
    **kwargs
):
    """
    PatchTST with reduced parameters for smaller datasets.
    Uses H=4, D=16, F=128 as recommended in paper for ILI, ETTh1, ETTh2.
    """
    return PatchTST(
        input_size=input_size,
        output_size=output_size,
        horizon=horizon,
        window_size=window_size,
        patch_len=16,
        stride=8,
        d_model=16,
        n_heads=4,
        n_layers=3,
        d_ff=128,
        dropout=0.2,
        exog_size=exog_size,
        **kwargs
    )


if __name__ == "__main__":
    # Test the model
    batch_size = 32
    window_size = 336
    horizon = 96
    num_nodes = 21
    input_features = 1
    
    # Create model
    model = PatchTST(
        input_size=input_features,
        output_size=input_features,
        horizon=horizon,
        window_size=window_size,
        patch_len=16,
        stride=8,
        d_model=128,
        n_heads=16,
        n_layers=3,
        d_ff=256,
        dropout=0.2
    )
    
    # Test with 4D input [B, T, N, F]
    x_4d = torch.randn(batch_size, window_size, num_nodes, input_features)
    out_4d = model(x_4d)
    print(f"Input shape (4D): {x_4d.shape}")
    print(f"Output shape (4D): {out_4d.shape}")
    print(f"Expected: [{batch_size}, {horizon}, {num_nodes}, {input_features}]")
    
    # Test with 3D input [B, T, F]
    x_3d = torch.randn(batch_size, window_size, input_features)
    out_3d = model(x_3d)
    print(f"\nInput shape (3D): {x_3d.shape}")
    print(f"Output shape (3D): {out_3d.shape}")
    print(f"Expected: [{batch_size}, {horizon}, {input_features}]")
    
    # Print model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Number of patches: {model.num_patches}")
