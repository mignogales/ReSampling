import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from tsl.nn.layers.graph_convs.diff_conv import DiffConv
from tsl.nn.models.base_model import BaseModel

class GRU(BaseModel):
    def __init__(self, 
                 input_size,
                 hidden_size,
                 output_size,
                 n_layers,
                 horizon,
                 dropout=0.1,
                 activation='relu',
                 exog_size=0,
                 n_nodes=None,
                 use_node_embeddings=False):
        
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.horizon = horizon
        self.dropout = dropout
        self.activation = activation
        self.exog_size = exog_size
        self.use_node_embeddings = use_node_embeddings
        
        self.input_encode = MLP(input_size + exog_size, hidden_size, hidden_size, dropout=dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True, dropout=dropout)
        self.fc = MLP(hidden_size, output_size * horizon, dropout=dropout)

        self.node_embeddings = nn.Parameter(torch.empty(n_nodes, hidden_size)) if use_node_embeddings else None
        if self.node_embeddings is not None:
            nn.init.xavier_uniform_(self.node_embeddings)

    def forward(self, x):
        """
        Forward pass of the GRUGCN model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_nodes, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_nodes, output_size).
        """

        # Encode input features
        x = self.input_encode(x)

        if self.use_node_embeddings:
            x = x + einops.repeat(self.node_embeddings, "n c -> b t n c", b=x.shape[0], t=x.shape[1])  # [batch, T_enc, N, hidden_size]

        # GRU layer
        x = self.gru(x)[0][:, -1]  # Get the output from the GRU

        # Fully connected layer
        x = self.fc(x)

        # rearrange output to match the expected shape
        x = einops.rearrange(x, 'b t -> b t 1')

        return x
                


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