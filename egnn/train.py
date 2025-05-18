import torch
import torch.nn as nn

from egnn.model import E3GNN
from utils.QM9DataLoader import load_qm9
from torch_geometric.loader import DataLoader

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(batch_size, lr = 1e-3, num_epochs = 100):

    dataset = load_qm9()

    for data in dataset:
        data.y = data.y[:, 7:8]

    n = len(dataset)
    train_dataset = dataset[:int(n * 0.8)]
    val_dataset = dataset[int(n * 0.8):int(n * 0.9)]
    test_dataset = dataset[int(n * 0.9):]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset)
    test_loader = DataLoader(test_dataset)

    atom_embedding = nn.Embedding(100, 1).to(device)


    model = E3GNN(irreps_in="1x0e", irreps_hidden="16x0e + 16x1o", irreps_out="1x0e").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    print("Training...")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            batch.x = atom_embedding(batch.z)
            
            optimizer.zero_grad()
            pred = model(batch)
            loss = loss_fn(pred, batch.y.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch} | Train Loss: {total_loss / len(train_loader):.6f}")


    model.eval()
    average_mse = 0.0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            batch.x = atom_embedding(batch.z)
            pred = model(batch)
            mse = loss_fn(pred, batch.y.squeeze())
            average_mse += mse.item()
    
    average_mse /= len(test_loader)
    print(f"Test MSE: {average_mse:.6f}")


if __name__ == "__main__":
    batch_size = 32
    lr = 1e-3
    num_epochs = 100
    train(batch_size, lr, num_epochs)