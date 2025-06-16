# useful_scripts.py

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from tqdm import tqdm
import random
from egnn.model import E3GNN
from utils.QM9DataLoader import load_qm9_with_energy

# --------- Config ---------
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --------- Utility Functions ---------
def load_structure_from_txt(file_path):
    """Load atomic structure from TXT file in format: x y z atom_number"""
    positions, atomic_numbers = [], []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            x, y, z = map(float, parts[:3])
            atomic_number = int(parts[3])
            positions.append([x, y, z])
            atomic_numbers.append(atomic_number)

    pos = torch.tensor(positions, dtype=torch.float)
    z = torch.tensor(atomic_numbers, dtype=torch.long)
    return pos, z


def save_structure_to_txt(filename, pos, atomic_numbers):
    """Save atomic structure to TXT file in format: x y z atom_number"""
    with open(filename, 'w') as f:
        for i in range(pos.shape[0]):
            x, y, z = pos[i].tolist()
            atom_num = atomic_numbers[i].item()
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {atom_num}\n")


def create_data_object(pos, z, atom_embedding, device):
    """Create PyG Data object from atomic positions and numbers"""
    pos = pos.to(device)
    z = z.to(device)
    x = atom_embedding(z)
    return Data(x=x, pos=pos, z=z)


def perturb_positions(pos, max_shift=0.5):
    """Add uniform noise to atomic positions"""
    noise = (torch.rand_like(pos) - 0.5) * 2 * max_shift
    return pos + noise


# --------- Core Testing Functions ---------
def test_single_example(model_path, original_file, noise_level=0.5):
    """Test if model prefers original or perturbed structure"""
    pos_orig, z = load_structure_from_txt(original_file)
    pos_mod = perturb_positions(pos_orig.clone(), noise_level=noise_level)

    atom_embedding = nn.Embedding(100, 16).to(device)

    data_orig = create_data_object(pos_orig, z, atom_embedding, device)
    data_mod = create_data_object(pos_mod, z, atom_embedding, device)

    model = E3GNN("16x0e", "16x0e + 16x1o", "1x0e").to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    loader = DataLoader([data_orig, data_mod], batch_size=2)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch).squeeze()
            return pred[0] < pred[1]  # True if original has lower energy


def test_one_molecule(model_path, file_path="../original_txt/structure_1.txt"):
    """Test a single molecule and save predictions and structures"""
    pos, z = load_structure_from_txt(file_path)
    atom_embedding = nn.Embedding(100, 16).to(device)
    original_data = create_data_object(pos, z, atom_embedding, device)

    model = E3GNN("16x0e", "16x0e + 16x1o", "1x0e").to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    modified_pos = perturb_positions(original_data.pos.clone(), max_shift=0.5)
    save_structure_to_txt("original_structure.txt", original_data.pos, z)
    save_structure_to_txt("modified_structure.txt", modified_pos, z)

    modified_data = create_data_object(modified_pos, z, atom_embedding, device)
    loader = DataLoader([original_data, modified_data], batch_size=2)

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch).squeeze()
            print(f"Original energy prediction: {pred[0].item():.12f}")
            print(f"Modified energy prediction: {pred[1].item():.12f}")


def test_noise_sweep(model_path, original_dir, noise_levels=None, n_samples=500):
    """Test model accuracy over a range of noise levels"""
    if noise_levels is None:
        noise_levels = [100, 50, 25, 10, 5, 2.5, 1.25, 1, 0.5, 0.25, 0.1,
                        0.05, 0.025, 0.01, 0.005, 0.0025, 0.001, 0.0005, 0.00025, 0.0001]

    for noise in noise_levels:
        correct = 0
        for i in range(n_samples):
            try:
                file = f"{original_dir}/structure_{i}.txt"
                if test_single_example(model_path, file, noise_level=noise):
                    correct += 1
            except Exception as e:
                print(f"{i}: error - {e}")
        print(f"Accuracy for noise {noise:.5f} Å: {(correct / n_samples) * 100:.2f}%")


def test_qm9_generalization(model_path, load_qm9_with_energy, n_perturb=5):
    """Evaluate model ability to select lowest-energy structure among perturbations"""
    dataset = load_qm9_with_energy()
    _, _, test_dataset = random_split(dataset, [0, 0, len(dataset)])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    atom_embedding = nn.Embedding(100, 16).to(device)
    model = E3GNN("16x0e", "16x0e + 16x1o", "1x0e").to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    correct, total = 0, 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing with perturbations"):
            batch = batch.to(device)
            original_energy = batch.y.item()
            original_pos = batch.pos.clone()
            energies = [original_energy]

            for _ in range(n_perturb):
                new_pos = perturb_positions(original_pos)
                new_batch = batch.clone()
                new_batch.pos = new_pos
                new_batch.x = atom_embedding(new_batch.z)
                new_batch.batch = torch.zeros_like(new_batch.z)
                pred_energy = model(new_batch).item()
                energies.append(pred_energy)

            if torch.tensor(energies).argmin().item() == 0:
                correct += 1
            total += 1

            if total % 1000 == 0:
                print(f"Progress: {total}, Accuracy: {(correct / total) * 100:.2f}%")

    print(f"Final Accuracy: {(correct / total) * 100:.2f}%")


# --------- Entry Point ---------
if __name__ == "__main__":
    model_path = "fine_tuned_model_weak.pth"

    # Example: test single molecule
    test_one_molecule(model_path)

    # Example: sweep over noise levels
    # test_noise_sweep(model_path, original_dir="../original_txt")

    # Example: QM9 generalization test
    # from utils.QM9DataLoader import load_qm9_with_energy
    # test_qm9_generalization(model_path, load_qm9_with_energy)
