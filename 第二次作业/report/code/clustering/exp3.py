import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score

# 生成非凸数据集（两个半月形）
X, y = make_moons(n_samples=300, noise=0.1, random_state=42)

# 谱聚类参数设置
affinities = ['nearest_neighbors', 'rbf']


def plot_clusters(X, y_true, spectral_labels, affinity):
    ari_spectral = adjusted_rand_score(y_true, spectral_labels)

    plt.figure(figsize=(6, 6))
    plt.scatter(X[:, 0], X[:, 1], c=spectral_labels, cmap='viridis', marker='.')
    plt.title(f'Spectral Clustering\nAffinity: {affinity}\nARI: {ari_spectral:.2f}')
    plt.show()


# 对不同的相似度度量方式进行谱聚类
for affinity in affinities:
    spectral = SpectralClustering(n_clusters=2, affinity=affinity, random_state=42)
    spectral_labels = spectral.fit_predict(X)
    plot_clusters(X, y, spectral_labels, affinity)
