from torch_geometric.datasets import MD17
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import torch
from egnn.model import E3GNN
import torch.nn as nn
import tqdm
from openqdc.datasets import GEOM

def test_md17(path):
    model = E3GNN(irreps_in="16x0e", irreps_hidden="16x0e + 16x1o", irreps_out="1x0e")
    model.load_state_dict(torch.load(path))
    model.eval()
    
    dataset = MD17(root='./md17', name='aspirin')  # You can choose 'ethanol', 'malonaldehyde', etc.
    
    atom_embedding = nn.Embedding(100, 16)
    loss_fn = torch.nn.MSELoss()

    n = 1000
    dataset = dataset[:n]
    preprocessed_data = []
    for data in dataset:
        energhy = data.energy  # Add energy as y
        # conver to Hartree energy
        data.y = energhy.unsqueeze(0) / 27.2114

        data.x = atom_embedding(data.z)  # Add atom embedding as x
        preprocessed_data.append(data)
    test_loader = DataLoader(preprocessed_data, batch_size=32)

    average_mse = 0.0
    with torch.no_grad():
        for batch in tqdm.tqdm(test_loader, desc="Testing"):
            print(f"Batch: {batch}")
            print(batch.y)
            print(batch.z)
            print(batch.pos)
            print(batch.batch)
            pred = model(batch)
            mse = loss_fn(pred, batch.y.squeeze())
            print(f"Pred {pred}, Target {batch.y.squeeze()}")
            average_mse += mse.item()
    
    average_mse /= len(test_loader)
    print(f"Test MSE: {average_mse:.6f}")


if __name__ == "__main__":
    test_md17("egnn_model_mse=1.pth")  # Path to the saved model