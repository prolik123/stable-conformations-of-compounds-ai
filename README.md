# Stable Conformations of Compounds AI

This repository contains scripts, and models for predicting the most stable conformations of compounds using AI techniques. The project is designed to help chemists and researchers quickly identify the most stable structures of various chemical compounds.

## Dependencies
To run the scripts in this repository, you will need the following Python packages:
- 'torch'
- 'torchvision'
- 'numpy'
- 'pyvista'
- 'torch_geometric'
- 'egnn'
- 'tqdm'
- 'matplotlib'
- 'collections'

## Model Overview
The model is based on a graph neural network (GNN) architecture that processes molecular structures represented as graphs. The GNN learns to predict the total energy of a compound based on its conformation, allowing it to identify the most stable structure.

## Datasets
The main training loop uses the QM9 dataset, but scripts for using MD17, and ANI-1ccx datasets are also provided. These datasets contain molecular structures and their corresponding energies, which are used to train the model.

## Usage
The code is not perfect, and usually the parameters for a model are hardcoded, and need to be changed manually. 

The QM9 training loop can be run with the following command:
```bash
python -m egnn.train
```
The model will save the best performing model to a file named `eggn_model.pth` in the current directory.
You can also provide a checkpoint file to continue training from a previous state.

The ANI-1ccx training loop can be run with the following command:
```bash
python -m egnn.ani
```

To test the model on MD17 dataset, you can use the following command:
```bash
python -m egnn.test
```

Also methods for data visulization are provided in the `Datasets Analysis` folder. 

Some utility scripts:
- `clf_txt.ipynb`: For changing the clf file format to a text file.
- `wizualizacja.py`: For visualizing one compound.
- `random_shift.py`: For shifting the coordinates of a compound randomly.
- `usefull_test_scripts.py`: Contains various utility functions for testing the model.

## Results
The model 1 was able to achieve a mean squered error (MSE) of 20 eV on the QM9 dataset. The model is present in the `eggn_model_20.pth` file, which can be used for further predictions or analysis.

The model 2 which is a fine tuned version of the model 1 on the ANI-1ccx, is available in the `fine_tuned_model_weak.pth` file. This model achieved worse results on both datasets, but it is more general and can be used for other datasets as well.

For generating data with shifted coordinates, use the `random_shift.py` script. And then feed the data to the `test` method in the `eggn.train` file.

CIF files that were used can be found in the `original_cif` folder. And modules with shifted coordinates in the `modified_txt` folder.

## Contributors

- Rafał Kajca
- Mateusz Hurkała
- Andrzej Sala
