import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score


def plot_clusters(X, y_true, kmeans_labels, spectral_labels, dataset_name):
    ari_kmeans = adjusted_rand_score(y_true, kmeans_labels)
    ari_spectral = adjusted_rand_score(y_true, spectral_labels)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis', marker='.')
    plt.title(f'K-means Clustering on {dataset_name}\nARI: {ari_kmeans:.2f}')

    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=spectral_labels, cmap='viridis', marker='.')
    plt.title(f'Spectral Clustering on {dataset_name}\nARI: {ari_spectral:.2f}')

    plt.show()


# 数据集1：两个半月形
X_moons, y_moons = make_moons(n_samples=300, noise=0.1, random_state=42)
kmeans_moons = KMeans(n_clusters=2, random_state=42)
kmeans_labels_moons = kmeans_moons.fit_predict(X_moons)
spectral_moons = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=42)
spectral_labels_moons = spectral_moons.fit_predict(X_moons)
plot_clusters(X_moons, y_moons, kmeans_labels_moons, spectral_labels_moons, 'Moons')

# 数据集2：两个圆环
X_circles, y_circles = make_circles(n_samples=300, factor=0.5, noise=0.05, random_state=42)
kmeans_circles = KMeans(n_clusters=2, random_state=42)
kmeans_labels_circles = kmeans_circles.fit_predict(X_circles)
spectral_circles = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=42)
spectral_labels_circles = spectral_circles.fit_predict(X_circles)
plot_clusters(X_circles, y_circles, kmeans_labels_circles, spectral_labels_circles, 'Circles')

# 数据集3：高斯混合
X_blobs, y_blobs = make_blobs(n_samples=300, centers=[(-2, -2), (2, 2)], cluster_std=1.5, random_state=42)
kmeans_blobs = KMeans(n_clusters=2, random_state=42)
kmeans_labels_blobs = kmeans_blobs.fit_predict(X_blobs)
spectral_blobs = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=42)
spectral_labels_blobs = spectral_blobs.fit_predict(X_blobs)
plot_clusters(X_blobs, y_blobs, kmeans_labels_blobs, spectral_labels_blobs, 'Blobs')
