import torch.nn as nn
from tsl.nn.models.base_model import BaseModel
import einops

class SimpleTemporalMLP(BaseModel):
    def __init__(self,
                 input_size,
                 output_size,
                 horizon,
                 window_size,
                 hidden_size=64,
                 dropout=0.0,
                 exog_size=0,
                 **kwargs):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.horizon = horizon
        self.window_size = window_size

        self.encoder = nn.Sequential(
            nn.Linear(input_size + exog_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.head = nn.Linear(hidden_size * window_size, output_size * horizon)

    def forward(self, x):
        """
        x: [B, T, N, F]
        returns: [B, horizon, N, output_size]
        """

        # Apply feature MLP independently at each time step and node
        original_shape = x.shape
        if len(original_shape) == 3:
            x = x.unsqueeze(2)  # [B, T, 1, F]
        x = self.encoder(x)          # [B, T, 1, H]

        x = einops.rearrange(x, 'b t n h -> b n (t h)')  # [B*N, T*H]

        x = self.head(x)             # [B, N, horizon * output_size]

        if len(original_shape) == 3:
            x = einops.rearrange(x, 'b 1 (h o) -> b h o', h=self.horizon, o=self.output_size)  # [B, N, horizon, output_size]
        else:
            x = einops.rearrange(x, 'b n (h o) -> b h n o', h=self.horizon, o=self.output_size)  # [B, horizon, N, output_size]
        
        return x
