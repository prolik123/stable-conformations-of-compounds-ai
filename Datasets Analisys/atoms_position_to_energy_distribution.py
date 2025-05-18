import torch
import matplotlib.pyplot as plt
from utils.QM9DataLoader import load_qm9

def _compute_distances(data):
    pos = data.pos
    distances = torch.cdist(pos, pos, p=2)
    # Considering only upper diagonal triangle
    i, j = torch.triu_indices(distances.size(0), distances.size(1), offset=1)
    return distances[i, j]

def compute_energy_distribution(dataset, limit = 100):
    bond_lengths = []
    energies = []
    energies_zero_kelvin = []
    for i in range(min(limit, len(dataset))):
        data = dataset[i]
        d = _compute_distances(data)
        bond_lengths.append(d.mean().item())
        energies.append(data.y[0][8].item()) # 8 -> Energy at 0 C
        energies_zero_kelvin.append(data.y[0][7].item()) # 7 -> Energy at 0 K
    return bond_lengths, energies, energies_zero_kelvin

def plot_energy_distribution(bond_lengths, energies):
    plt.scatter(bond_lengths, energies, alpha=0.5, color='blue')
    plt.title("Zależność długości wiązań od energii cząsteczki (U)")
    plt.xlabel("Średnia odległość międzyatomowa [Å]")
    plt.ylabel("Energia całkowita (U) przy 298.15 K")
    plt.grid(True)
    plt.show()


def plot_0_k_energy_distribution(bond_lengths, energies_zero_kelvin):
    plt.scatter(bond_lengths, energies_zero_kelvin, alpha=0.5, color='green')
    plt.title("Zależność długości wiązań od energii cząsteczki (U0)")
    plt.xlabel("Średnia odległość międzyatomowa [Å]")
    plt.ylabel("Energia całkowita (U0) przy 0 K")
    plt.grid(True)
    plt.show()

def analise_and_plot(dataset, limit = 100):
    bond_lengths, energies, energies_zero_kelvin = compute_energy_distribution(dataset, limit)
    plot_energy_distribution(bond_lengths, energies)
    plot_0_k_energy_distribution(bond_lengths, energies_zero_kelvin)

if __name__ == "__main__":
    dataset = load_qm9()
    analise_and_plot(dataset)