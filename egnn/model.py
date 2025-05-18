import torch
from torch.nn import Module
from torch_geometric.nn import radius_graph
from torch_geometric.data import Data
from e3nn.o3 import Irreps
from e3nn.o3 import FullyConnectedTensorProduct
from e3nn.nn import Gate
from e3nn.o3 import Linear


class E3GNN(Module):
    def __init__(self, irreps_in="1x0e", irreps_hidden="16x0e + 16x1o", irreps_out="1x0e", 
                 num_layers=3, num_neighbors=12, cutoff=5.0):
        super().__init__()
        self.num_layers = num_layers
        self.num_neighbors = num_neighbors
        self.cutoff = cutoff

        self.irreps_in = Irreps(irreps_in)
        self.irreps_hidden = Irreps(irreps_hidden)
        self.irreps_out = Irreps(irreps_out)

        self.input_proj = Linear(self.irreps_in, self.irreps_hidden)

        self.layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            layer = torch.nn.Sequential(
                FullyConnectedTensorProduct(self.irreps_hidden, self.irreps_hidden, self.irreps_hidden),
                Gate(self.irreps_hidden)
            )
            self.layers.append(layer)

        self.readout = Linear(self.irreps_hidden, self.irreps_out)

    def forward(self, data: Data):
        pos = data.pos
        batch = data.batch
        x = data.x 

        edge_index = radius_graph(pos, r=self.cutoff, batch=batch, max_num_neighbors=self.num_neighbors)
        row, col = edge_index

        x = self.input_proj(x)

        for layer in self.layers:
            messages = layer[0](x[row], x[col])
            agg = torch.zeros_like(x)
            agg.index_add_(0, row, messages)
            x = layer[1](agg)

        out = self.readout(x)

        energy = torch.zeros(batch.max() + 1, device=x.device)
        energy.index_add_(0, batch, out.squeeze(-1)) 

        return energy