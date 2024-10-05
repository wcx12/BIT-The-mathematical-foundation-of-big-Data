import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score

# 生成非凸数据集（两个半月形）
X, y = make_moons(n_samples=300, noise=0.1, random_state=42)

# 谱聚类参数设置
neighbors_list = [2, 5, 10, 20]


def plot_clusters(X, y_true, spectral_labels, n_neighbors):
    ari_spectral = adjusted_rand_score(y_true, spectral_labels)

    plt.figure(figsize=(6, 6))
    plt.scatter(X[:, 0], X[:, 1], c=spectral_labels, cmap='viridis', marker='.')
    plt.title(f'Spectral Clustering\nn_neighbors: {n_neighbors}\nARI: {ari_spectral:.2f}')
    plt.show()


# 对不同的n_neighbors进行谱聚类
for n_neighbors in neighbors_list:
    spectral = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', n_neighbors=n_neighbors, random_state=42)
    spectral_labels = spectral.fit_predict(X)
    plot_clusters(X, y, spectral_labels, n_neighbors)
