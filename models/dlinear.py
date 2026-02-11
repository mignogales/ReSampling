import torch
import torch.nn as nn
from tsl.nn.models.base_model import BaseModel
import einops

class MovingAvg(nn.Module):
    """
    Moving average block to extract the trend-cyclical component.
    Matches the implementation used in Autoformer and FEDformer.
    """
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, self.kernel_size // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class SeriesDecomposition(nn.Module):
    """
    Series decomposition block.
    """
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class LTSFDLinear(BaseModel):
    """
    Implementation of the DLinear model.
    It decomposes the raw input into trend and seasonal components,
    applies linear layers to each, and sums them for the final prediction.
    """
    def __init__(self,
                 input_size,
                 output_size,
                 horizon,
                 window_size,
                 moving_avg_kernel=25, # Default kernel size from the paper
                 **kwargs):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.horizon = horizon
        self.window_size = window_size

        # Decomposition block
        self.decomposition = SeriesDecomposition(moving_avg_kernel)
        
        # Specialized linear layers for each component
        # These are applied along the temporal axis (window_size -> horizon)
        self.linear_seasonal = nn.Linear(self.window_size, self.horizon)
        self.linear_trend = nn.Linear(self.window_size, self.horizon)

    def forward(self, x):
        """
        x: [B, L, N, F]
        returns: [B, T, N, output_size]
        """
        # 1. Flatten spatial/feature dimensions to treat each variate independently
        # Input shape for decomp: [Batch, Time, Variates]
        # We treat (Nodes * Features) as the variate dimension
        original_shape = x.shape
        if len(original_shape) == 3:
            x = x.unsqueeze(2)  # [B, T, 1, F]
            original_shape = x.shape
        x = einops.rearrange(x, 'b l n f -> b l (n f)')

        # 2. Decompose into seasonal and trend components
        seasonal_init, trend_init = self.decomposition(x) # [B, L, NF]

        # 3. Apply linear layers along the temporal dimension
        # Rearrange to [B, NF, L] for the Linear layer [L -> T]
        seasonal_init = seasonal_init.transpose(1, 2)
        trend_init = trend_init.transpose(1, 2)

        seasonal_output = self.linear_seasonal(seasonal_init) # [B, NF, T]
        trend_output = self.linear_trend(trend_init)          # [B, NF, T]

        # 4. Sum components and reshape back to standard output
        x = seasonal_output + trend_output # [B, NF, T]
        
        # Rearrange back to [B, T, N, F]
        x = einops.rearrange(x, 'b (1 f) t -> b t f', f=original_shape[3])

        return x[..., :self.output_size]