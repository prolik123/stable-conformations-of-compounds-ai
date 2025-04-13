import torch
import matplotlib.pyplot as plt
from utils.QM9DataLoader import load_qm9

def _distance_from_centroid(data):
    pos = data.pos
    centroid = pos.mean(dim=0)
    return torch.norm(pos - centroid, dim=1)

def count_avg_dists(data, limit = 100):
    return [_distance_from_centroid(data[i]).mean().item() for i in range(min(limit, len(data)))]

def plot_centroid_distance(avg_dists):
    plt.hist(avg_dists, bins=100, color='coral', edgecolor='black')
    plt.title("Średnia odległość atomów od środka masy")
    plt.xlabel("Odległość [Å]")
    plt.ylabel("Liczba cząsteczek")
    plt.show()

def analise_and_plot(dataset, limit = 100):
    avg_dists = count_avg_dists(dataset, limit)
    plot_centroid_distance(avg_dists)

if __name__ == "__main__":
    dataset = load_qm9()
    analise_and_plot(dataset)
