from einops.layers.torch import Rearrange
import torch.nn as nn
from tsl.nn.blocks.encoders import RNN
from tsl.nn.layers import DiffConv, NodeEmbedding


class TimeThenSpaceModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        n_nodes: int,
        horizon: int,
        window: int,
        stride: int,
        hidden_size: int = 32,
        rnn_layers: int = 1,
        gnn_kernel: int = 2,
        **kwargs,
    ):
        super(TimeThenSpaceModel, self).__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.node_embeddings = NodeEmbedding(n_nodes, hidden_size)
        self.time_nn = RNN(
            input_size=hidden_size,
            hidden_size=hidden_size,
            n_layers=rnn_layers,
            cell="gru",
            return_only_last_state=True,
        )
        self.space_nn = DiffConv(in_channels=hidden_size, out_channels=hidden_size, k=gnn_kernel)
        self.decoder = nn.Linear(hidden_size, input_size * horizon)
        self.rearrange = Rearrange("b n (t f) -> b t n f", t=horizon)

    def forward(self, x, edge_index, edge_weight):
        # x has shape [batch, window, nodes, features]
        x_enc = self.encoder(x)
        x_emb = x_enc + self.node_embeddings()
        # h will have shape [batch, nodes, hidden_size]
        h = self.time_nn(x_emb)
        # z will have shape [batch, nodes, hidden_size]
        z = self.space_nn(h, edge_index, edge_weight)
        # x_out will have shape [batch, nodes, horizon * features]
        x_out = self.decoder(z)
        # x_horizon will have shape [batch, horizon, nodes, features]
        x_horizon = self.rearrange(x_out)
        return x_horizon
