import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from utils.GEOMDataLoader import load_geom_sample

ATOM_COLORS = {
    1: 'pink',   # H
    6: 'black',   # C
    7: 'blue',    # N
    8: 'red',     # O
    9: 'green',   # F
}

ATOM_LABELS = {
    1: 'H',
    6: 'C',
    7: 'N',
    8: 'O',
    9: 'F',
}

def plot_conformer_3d(Z, coords, title="Molecule"):
    coords = np.array(coords)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)

    for z, (x, y, z_) in zip(Z, coords):
        color = ATOM_COLORS.get(z, 'gray')
        ax.scatter(x, y, z_, color=color, s=100, edgecolors='k')
        ax.text(x, y, z_, ATOM_LABELS.get(z, str(z)), fontsize=8)

    ax.set_xlabel("X [Å]")
    ax.set_ylabel("Y [Å]")
    ax.set_zlabel("Z [Å]")
    ax.view_init(elev=20, azim=30)

    # Create legend
    legend_patches = [
        mpatches.Patch(color=color, label=ATOM_LABELS.get(z, str(z)))
        for z, color in ATOM_COLORS.items()
    ]
    ax.legend(handles=legend_patches, loc='upper right')

    plt.tight_layout()
    plt.show()


def view_conformers(geom_data):
    it = iter(geom_data)
    next(it)
    key = next(it)
    print(f"Result for: {key}")
    for i, mol_data in enumerate(geom_data[key]["conformers"]):
        coords = [z[1:] for z in mol_data['xyz']]
        energies = mol_data['totalenergy']
        Z = [int(z[0]) for z in mol_data['xyz']]
        plot_conformer_3d(Z, coords, f"Atomy dla {i}-ej konformacji [Energia całkowita] - {energies}")

def analyse_and_polt():
    crude_dataset, features_dataset = load_geom_sample()
    view_conformers(crude_dataset)

if __name__ == "__main__":
    analyse_and_polt()