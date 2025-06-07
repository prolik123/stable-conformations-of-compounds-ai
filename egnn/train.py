import torch
import torch.nn as nn

from egnn.model import E3GNN
from utils.QM9DataLoader import load_qm9_with_energy
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from torch.utils.data import random_split
from torch.utils.data import Subset

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(batch_size, lr = 1e-3, num_epochs = 20, samples = 10000000):

    dataset = load_qm9_with_energy()

    #for data in dataset:
    #    data.y = data.y[:, 12].unsqueeze(0)

    n = len(dataset)
    n = min(n, samples)
    subset_indices = list(range(n))
    dataset = Subset(dataset, subset_indices)
    train_len = int(n * 0.8)
    val_len = int(n * 0.1)
    test_len = n - train_len - val_len
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_len, val_len, test_len])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset)
    test_loader = DataLoader(test_dataset)
    print(f"Dataset sizes: Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    atom_embedding = nn.Embedding(100, 16).to(device)


    model = E3GNN(irreps_in="16x0e", irreps_hidden="16x0e + 16x1o", irreps_out="1x0e").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    step_size = num_epochs // 3 if num_epochs > 3 else 1
    gamma = lr ** (1 / step_size)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=num_epochs, gamma=gamma)
    loss_fn = torch.nn.MSELoss()

    print("Training...")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            batch = batch.to(device)
            batch.x = atom_embedding(batch.z)
            #print(f"{batch[0]}, {batch[0].pos}")
            optimizer.zero_grad()
            pred = model(batch)
            #print(f"{batch}")
            #print(f"Batch size: {batch.num_graphs}, Pred shape: {pred.shape}, Target shape: {batch.y.shape}")
            #print(f"pred: {pred.squeeze()}, target: {batch.y.squeeze()}")
            loss = loss_fn(pred.squeeze(), batch.y.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        
        # validate
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for batch in val_loader:
                batch = batch.to(device)
                batch.x = atom_embedding(batch.z)
                pred = model(batch)
                loss = loss_fn(pred, batch.y.squeeze())
                val_loss += loss.item()
            print(f"Epoch {epoch} | Train Loss: {total_loss / len(train_loader):.6f}, Val Loss: {val_loss / len(val_loader):.6f}")


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

    # Save the model
    torch.save(model.state_dict(), "egnn_model.pth")


def test():
    model = E3GNN(irreps_in="16x0e", irreps_hidden="16x0e + 16x1o", irreps_out="1x0e")
    model.load_state_dict(torch.load("egnn_model.pth"))
    model.eval()

    dataset = load_qm9_with_energy()
    sample = dataset[5]
    sample.batch = torch.zeros(sample.num_nodes, dtype=torch.long)
    sample.x = nn.Embedding(100, 16)(sample.z)

    with torch.no_grad():
        pred = model(sample)
        mse = nn.MSELoss()(pred, sample.y.unsqueeze(0))
        print(f"Predicted energy: {pred.item()}, Actual energy: {sample.y.item()}, MSE: {mse.item()}")

if __name__ == "__main__":
    batch_size = 15
    lr = 1e-3
    num_epochs = 5
    samples = 100000000000 # inf
    train(batch_size, lr, num_epochs, samples)
    #test()