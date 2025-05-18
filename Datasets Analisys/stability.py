import torch
import matplotlib.pyplot as plt
from utils.QM9DataLoader import load_qm9

def analyse_stability(dataset, limit=100):
    energies_u0 = []
    num_atoms = []
    avg_bond_lengths = []
    energy_per_atom = []

    for i in range(limit):
        data = dataset[i]
        pos = data.pos
        n_atoms = data.num_nodes
        num_atoms.append(n_atoms)

        dists = torch.cdist(pos, pos)
        i_triu, j_triu = torch.triu_indices(dists.size(0), dists.size(1), offset=1)
        mean_dist = dists[i_triu, j_triu].mean().item()
        avg_bond_lengths.append(mean_dist)

        u0 = data.y[0][7].item()
        energies_u0.append(u0)
        energy_per_atom.append(u0 / n_atoms)

    return num_atoms, avg_bond_lengths, energy_per_atom

def plot_stability(num_atoms, energy_per_atom, avg_bond_lengths):
    plt.figure(figsize=(6, 4))
    plt.scatter(num_atoms, energy_per_atom, alpha=0.5)
    plt.xlabel("Liczba atomów")
    plt.ylabel("Energia U0 na atom [Hartree]")
    plt.title("Energia na atom względem wielkości cząsteczki")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.scatter(avg_bond_lengths, energy_per_atom, alpha=0.5)
    plt.xlabel("Średnia odległość międzyatomowa [Å]")
    plt.ylabel("Energia U0 na atom [Hartree]")
    plt.title("Energia a zwartość struktury")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.hist(energy_per_atom, bins=50, color='skyblue', edgecolor='k')
    plt.title("Rozkład energii na atom (U0)")
    plt.xlabel("Energia U0 na atom [Hartree]")
    plt.ylabel("Liczba cząsteczek")
    plt.grid(True)
    plt.show()

def analyse_and_plot(dataset, limit = 100):
    num_atoms, avg_bond_lengths, energy_per_atom = analyse_stability(dataset, limit)
    plot_stability(num_atoms, avg_bond_lengths, energy_per_atom)

if __name__ == '__main__':
    dataset = load_qm9()
    analyse_and_plot(dataset)
