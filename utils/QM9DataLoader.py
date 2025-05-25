import torch
from torch_geometric.datasets import QM9
from torch_geometric.transforms import BaseTransform


def _add_all_edges(data):
    num_nodes = data.num_nodes
    row = torch.arange(num_nodes).repeat(num_nodes)
    col = torch.arange(num_nodes).repeat_interleave(num_nodes)
    edge_index = torch.stack([row, col], dim=0)

    # (self-loops)
    mask = row != col
    data.edge_index = edge_index[:, mask]
    return data

def load_qm9():
    return QM9(root='../Datasets/QM9', transform=_add_all_edges)

# labels distribution https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.QM9.html

class QM9EnergyTransform(BaseTransform):
    def __call__(self, data):
        # Assuming the energy is in the 12th column of the target tensor
        data.y = data.y[:, 12].unsqueeze(1)  # Keep it as a 2D tensor
        return data



def load_qm9_with_energy():
    return QM9(root='../Datasets/QM9', transform=QM9EnergyTransform(), pre_transform=_add_all_edges)




if __name__ == '__main__':
    load_qm9()