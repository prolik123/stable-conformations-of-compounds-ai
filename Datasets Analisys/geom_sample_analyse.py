from rdkit import Chem
import py3Dmol
import numpy as np
from utils.GEOMDataLoader import load_geom_sample

def geom_to_rdkit_mol(smiles, atomic_numbers, coords):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    conf = Chem.Conformer(len(atomic_numbers))
    for i, pos in enumerate(coords):
        conf.SetAtomPosition(i, pos.tolist())

    mol.AddConformer(conf, assignId=True)
    return mol

def view_conformers(geom_data):
    key = next(iter(geom_data))
    # it does not work now :(
"""
    min_idx = np.argmin(energies)
    max_idx = np.argmax(energies)

    mol_min = geom_to_rdkit_mol(smiles, Z, coords[min_idx])
    mol_max = geom_to_rdkit_mol(smiles, Z, coords[max_idx])

    # Create viewer
    viewer = py3Dmol.view(width=800, height=400)
    viewer.addModel(Chem.MolToMolBlock(mol_min), 'mol')
    viewer.setStyle({'stick': {}})
    viewer.setBackgroundColor('0xeeeeee')
    viewer.zoomTo()
    viewer.show()

    print(f"SMILES: {smiles}")
    print(f"Min energy: {energies[min_idx]:.2f} kcal/mol")
    print(f"Max energy: {energies[max_idx]:.2f} kcal/
    """


if __name__ == "__main__":
    crude_dataset, features_dataset = load_geom_sample()
    view_conformers(features_dataset)