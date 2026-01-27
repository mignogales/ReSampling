import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from tsl.nn.models.base_model import BaseModel

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64, n_layers=2, dropout=0.1):
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden_size, output_size))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class MixerBlock(nn.Module):
    """
    Standard Mixer Block containing Time-Mixing and Channel-Mixing layers.
    """
    def __init__(self, window_size, hidden_size, dropout=0.1):
        super().__init__()
        
        # --- Time Mixing ---
        self.norm_time = nn.LayerNorm(hidden_size)
        # We process the time dimension using an MLP. 
        # Input to MLP will be [Batch, Features, Time], so input_size=window_size
        self.time_mix = MLP(input_size=window_size, 
                            output_size=window_size, 
                            hidden_size=window_size * 2, # Expansion factor usually 2-4x
                            n_layers=2, 
                            dropout=dropout)
        
        # --- Channel (Feature) Mixing ---
        self.norm_feature = nn.LayerNorm(hidden_size)
        self.feature_mix = MLP(input_size=hidden_size, 
                               output_size=hidden_size, 
                               hidden_size=hidden_size * 2, 
                               n_layers=2, 
                               dropout=dropout)

    def forward(self, x):
        # x shape: [Batch, Time, Hidden]
        
        # 1. Time Mixing: Acts on the Time dimension
        # Norm -> Transpose -> MLP(Time) -> Transpose -> Add Residual
        y = self.norm_time(x)
        y = einops.rearrange(y, 'b t h -> b h t') # [Batch, Hidden, Time]
        y = self.time_mix(y)
        y = einops.rearrange(y, 'b h t -> b t h') # Back to [Batch, Time, Hidden]
        x = x + y # Residual connection

        # 2. Channel Mixing: Acts on the Hidden dimension
        # Norm -> MLP(Hidden) -> Add Residual
        y = self.norm_feature(x)
        y = self.feature_mix(y)
        x = x + y # Residual connection
        
        return x

class DMixer(BaseModel):
    def __init__(self, 
                 input_size,
                 hidden_size,
                 output_size,
                 n_layers,
                 horizon,
                 window_size, # Required for Mixer to know sequence length
                 dropout=0.1,
                 activation='relu',
                 exog_size=0,
                 n_nodes=None): # Kept for compatibility, but unused for spatial mixing
        
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.horizon = horizon
        self.window_size = window_size
        self.dropout = dropout
        self.activation = activation
        
        # Encoder: Projects raw inputs + exog to hidden space
        # [B, T, N, F] -> [B, T, N, H]
        self.input_encode = MLP(input_size + exog_size, hidden_size, hidden_size, dropout=dropout)

        # Mixer Blocks: The core backbone
        self.mixer_blocks = nn.ModuleList([
            MixerBlock(window_size=window_size, hidden_size=hidden_size, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Decoder / Forecast Head
        # Projects the latent representation to the prediction horizon
        # We flatten the time dimension or take the last step. 
        # Here we project from (Hidden) -> (Horizon * Output)
        self.head = MLP(hidden_size, output_size * horizon, dropout=dropout)

    def forward(self, x):
        """
        Forward pass of the DMixer model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, window_size, num_nodes, input_size).
                              Note: TSL standards usually place nodes at dim 2.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, horizon, num_nodes, output_size).
        """
        
        # 1. Encode input features
        # x: [Batch, Time, Nodes, Features]
        x = self.input_encode(x) 
        
        # 2. Prepare for Mixing (Skip Spatial / Graph components)
        # We treat every node as an independent sample in the batch.
        # [Batch, Time, Nodes, Hidden] -> [Batch * Nodes, Time, Hidden]
        b, t, h = x.shape

        # 3. Apply Mixer Blocks
        for block in self.mixer_blocks:
            x = block(x)
        
        # 4. Decoding / Forecasting
        # Common approach: Take the last time step representation or pool
        x = x[:, -1, :] # [Batch * Nodes, Hidden]
        
        # Project to output
        x = self.head(x) # [Batch * Nodes, Horizon * Output]

        # 5. Reshape to desired output format
        # [Batch * Nodes, Horizon * Output] -> [Batch, Horizon, Nodes, Output]
        x = einops.rearrange(x, 'b (t o) -> b t o', b=b, t=self.horizon)

        return x