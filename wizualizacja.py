import numpy as np
import pyvista as pv

# Kolory CPK dla przykładowych pierwiastków
CPK_COLORS = {
    1:  'white',      # H
    6:  'black',      # C
    7:  'blue',       # N
    8:  'red',        # O
    9:  'green',      # F
    16: 'yellow',     # S
}

# Promienie Van der Waalsa (w angstromach)
VDW_RADII = {
    1: 0.31,
    6: 0.76,
    7: 0.71,
    8: 0.66,
    9: 0.57,
    16: 1.05,
}

def load_xyz_file(filepath):
    atoms = []
    with open(filepath, 'r') as file:
        for line in file:
            if not line.strip(): continue
            x, y, z, z_atomic = map(float, line.strip().split())
            atoms.append((x, y, z, int(z_atomic)))
    return atoms

def visualize_molecule(atoms):
    plotter = pv.Plotter()
    for x, y, z, z_atom in atoms:
        color = CPK_COLORS.get(z_atom, 'gray')
        radius = VDW_RADII.get(z_atom, 0.8)
        sphere = pv.Sphere(radius=radius, center=(x, y, z), theta_resolution=32, phi_resolution=32)
        plotter.add_mesh(sphere, color=color, smooth_shading=True)

    plotter.show_grid()
    plotter.show()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Użycie: python wizualizuj.py plik.txt")
    else:
        atoms = load_xyz_file(sys.argv[1])
        visualize_molecule(atoms)