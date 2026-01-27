import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

# Fallback if TSL is not installed, effectively making this a standalone PyTorch model
try:
    from tsl.nn.models.base_model import BaseModel
except ImportError:
    BaseModel = nn.Module

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

class SeriesDecomp(nn.Module):
    """
    Decomposes a time series into Seasonality (Periodic) and Trend components.
    Operates on (Batch, Time, Features).
    """
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)
        self.kernel_size = kernel_size

    def forward(self, x):
        # x shape: [Batch, Time, Features]
        
        # Padding to keep time dimension unchanged after pooling
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x_pad = torch.cat([front, x, end], dim=1)
        
        # Permute to [Batch, Features, Time] for AvgPool1d
        x_pad = x_pad.permute(0, 2, 1)
        
        # Calculate Trend
        x_trend = self.moving_avg(x_pad)
        x_trend = x_trend.permute(0, 2, 1) # Back to [Batch, Time, Features]
        
        # Seasonality = Original - Trend
        x_seasonal = x - x_trend
        return x_seasonal, x_trend

class MixerBlock(nn.Module):
    """
    Mixes information across Time and Features (Channels).
    No spatial mixing occurs here.
    """
    def __init__(self, seq_len, hidden_size, dropout=0.1):
        super(MixerBlock, self).__init__()
        
        # 1. Time Mixing (Interacts across time steps)
        self.time_mixer = nn.Sequential(
            nn.Linear(seq_len, seq_len),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 2. Feature Mixing (Interacts across channels/features)
        self.feature_mixer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Input x: [Batch, Time, Features]
        
        # --- Time Mixing ---
        # Transpose to [Batch, Features, Time] so Linear applies to Time
        y = x.transpose(1, 2)
        y = self.time_mixer(y)
        y = y.transpose(1, 2)
        x = x + self.dropout(y)
        
        # --- Feature Mixing ---
        # Linear applies to last dim (Features) by default
        y = self.feature_mixer(x) 
        x = x + self.dropout(y)
        
        return self.norm(x)

class TimeMixer(BaseModel):
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 output_size, 
                 n_layers, 
                 horizon, 
                 seq_len,
                 down_sampling_layers=[1, 2], # Multi-scale factors
                 dropout=0.1):
        
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.horizon = horizon
        self.seq_len = seq_len
        self.down_sampling_layers = down_sampling_layers
        
        # Decomposition Block
        self.decomp = SeriesDecomp(kernel_size=25)
        
        # Components for each scale
        self.encoders = nn.ModuleList()
        self.projections = nn.ModuleList()
        self.embeddings = nn.ModuleList()

        for factor in self.down_sampling_layers:
            current_seq_len = seq_len // factor
            
            # Projects input features to hidden space
            self.embeddings.append(
                nn.Linear(input_size, hidden_size)
            )

            # Stack of Mixer Blocks
            encoder_layers = []
            for _ in range(n_layers):
                encoder_layers.append(
                    MixerBlock(current_seq_len, hidden_size, dropout)
                )
            self.encoders.append(nn.Sequential(*encoder_layers))
            
            # Projects (Time_Scale * Hidden) -> (Horizon * Output)
            # This is the "Decoder" part
            self.projections.append(
                MLP(current_seq_len * hidden_size, horizon * output_size, hidden_size, dropout=dropout)
            )

    def forward(self, x):
        """
        Args:
            x: Input tensor. Can be:
               - (Batch, Time, Features)
               - (Batch, Time, Nodes, Features)
        Returns:
            Output tensor. Matches input rank:
               - (Batch, Horizon, Features)
               - (Batch, Horizon, Nodes, Features)
        """
        # --- 1. Input Reshaping ---
        # Now x is always [Batch_Eff, Time, Features]
        
        batch_eff = x.shape[0]
        outputs_sum = torch.zeros(batch_eff, self.horizon * self.output_size).to(x.device)

        # --- 2. Multi-Scale Processing ---
        for i, factor in enumerate(self.down_sampling_layers):
            
            # A. Downsample (Pooling over Time)
            if factor > 1:
                # Permute to [Batch, Features, Time] for pooling
                x_scale = x.transpose(1, 2)
                x_scale = F.avg_pool1d(x_scale, kernel_size=factor, stride=factor)
                x_scale = x_scale.transpose(1, 2)
            else:
                x_scale = x

            # B. Decompose
            seasonal, trend = self.decomp(x_scale)
            
            # C. Embed
            enc_out = self.embeddings[i](seasonal) # [Batch, T_scale, Hidden]

            # D. Mix (Encoder)
            for layer in self.encoders[i]:
                enc_out = layer(enc_out)

            # E. Project (Decoder)
            # Flatten Time and Hidden for the MLP
            enc_out_flat = einops.rearrange(enc_out, 'b t h -> b (t h)')
            
            # Project to output size
            dec_out = self.projections[i](enc_out_flat) # [Batch, Horizon * Output]
            
            # Accumulate
            outputs_sum = outputs_sum + dec_out

        # --- 3. Reshape Output ---
        # Reshape flat output to [Batch, Horizon, Features]
        y = einops.rearrange(outputs_sum, 'b (h o) -> b h o', h=self.horizon)

        return y