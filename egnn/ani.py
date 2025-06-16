from load_atoms import load_dataset
import torch
from torch_geometric.data import Dataset, Data
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import torch.nn as nn
from egnn.model import E3GNN
from tqdm import tqdm

class ANILoadAtomsDataset(Dataset):
    def __init__(self, atoms_dataset):
        super().__init__()
        self.atoms = atoms_dataset

    def len(self):
        return len(self.atoms)

    def get(self, idx):
        atom = self.atoms[idx]
        pos = torch.tensor(atom.positions, dtype=torch.float)
        z = torch.tensor(atom.numbers, dtype=torch.long)
        y = torch.tensor([atom.info["energy"]], dtype=torch.float)

        return Data(pos=pos, z=z, y=energy)


atoms_dataset = load_dataset("ANI-1x")

pyg_dataset = ANILoadAtomsDataset(atoms_dataset)

n = 100
_, _, test_dataset = random_split(pyg_dataset, [0, 0, n])

test_loader = DataLoader(test_dataset, shuffle=True)

loader = DataLoader(pyg_dataset, batch_size=16, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

atom_embedding = nn.Embedding(100, 16).to(device)

path = "egnn_model_20.pth"

model = E3GNN(irreps_in="16x0e", irreps_hidden="16x0e + 16x1o", irreps_out="1x0e").to(device)
if path is not None:
    model.load_state_dict(torch.load(path))
    print(f"Loaded model from {path}")
loss_fn = torch.nn.MSELoss()

model.eval()
average_mse = 0.0
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        batch = batch.to(device)
        batch.x = atom_embedding(batch.z)
        pred = model(batch)
        print(f"batch size: {batch}, Pred shape: {batch.batch}")
        
        print(f"pred: {pred.squeeze()}, target: {batch.y.squeeze()}")
        mse = loss_fn(pred.squeeze(), batch.y.squeeze())
        average_mse += mse.item()
        break