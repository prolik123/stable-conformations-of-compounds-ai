import torch
from torch_geometric.datasets import QM9


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




if __name__ == '__main__':
    load_qm9()