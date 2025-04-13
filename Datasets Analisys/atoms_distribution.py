import torch
import matplotlib.pyplot as plt
from utils.QM9DataLoader import load_qm9

def compute_distances(data):
    pos = data.pos
    distances = torch.cdist(pos, pos, p=2)
    # Considering only upper diagonal triangle
    i, j = torch.triu_indices(distances.size(0), distances.size(1), offset=1)
    return distances[i, j]

# Samples from first 100 elements
def count_distances(data, limit = 100):
    all_distances = [compute_distances(data[i]) for i in range(min(limit, len(data)))]
    return torch.cat(all_distances).numpy()

def show_plot(distances):
    plt.hist(distances, bins=100, color='skyblue', edgecolor='black')
    plt.title("Histogram odległości międzyatomowych (XYZ)")
    plt.xlabel("Odległość [Å]")
    plt.ylabel("Liczba par atomów")
    plt.show()

def analise_and_plot(dataset, limit = 100):
    dists = count_distances(dataset, limit)
    show_plot(dists)

if __name__ == "__main__":
    dataset = load_qm9()
    analise_and_plot(dataset)

