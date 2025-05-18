import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from utils.QM9DataLoader import load_qm9
from utils import periodictable_utils

# magic fact from QM9 docs dataset[i].z is a list of atomic number for each atom

# O, C, H are analise by default
_default_atomic_numbers_to_analise = [8, 1 ,6]

def get_position_by_element_type(dataset, limit = 100):
    positions_by_element = defaultdict(list)
    for i in range(min(limit, len(dataset))):
        data = dataset[i]
        for atom_type, pos in zip(data.z.tolist(), data.pos.tolist()):
            positions_by_element[atom_type].append(pos)
    return positions_by_element

def plot_positions_by_element(positions_by_element, atomic_numbers = None):
    # Example for few choosen types of atoms

    if atomic_numbers is None:
        atomic_numbers = _default_atomic_numbers_to_analise
    for atom_number in atomic_numbers:
        atom_positions = np.array(positions_by_element[atom_number])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(atom_positions[:, 0], atom_positions[:, 1], atom_positions[:, 2], alpha=0.6)
        ax.set_title(f"Positions of {periodictable_utils.get_name_by_atomic_number(atom_number)} atoms" +
                     f"({periodictable_utils.get_symbol_by_atomic_number(atom_number)})")
        plt.show()

def analise_and_plot(dataset, limit = 100, atomic_numbers = None):
    positions = get_position_by_element_type(dataset, limit)
    plot_positions_by_element(positions, atomic_numbers)

if __name__ == "__main__":
    dataset = load_qm9()
    analise_and_plot(dataset)