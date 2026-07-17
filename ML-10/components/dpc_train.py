"""密度峰值聚类 (DPC, Rodriguez & Laio 2014) — Concrete 数据集（去标签）训练、评估与可视化"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score


"""
密度峰值聚类核心算法：计算 rho, delta, gamma 和最近更高密度点。

Parameters
----------
X : ndarray
    特征矩阵 (n_samples, n_features)。
dc_percent : float, optional
    截断距离百分比，默认 2.0。

Returns
-------
rho : ndarray
    局部密度。
delta : ndarray
    到更高密度点的最小距离。
gamma : ndarray
    rho * delta。
order : ndarray
    按 rho 降序排列的索引。
nearest_higher : ndarray
    每个点的最近更高密度点索引。
"""
def density_peaks(X, dc_percent=2.0):
    n = X.shape[0]
    nbrs = NearestNeighbors(n_neighbors=n).fit(X)
    distances, _ = nbrs.kneighbors(X)

    all_dist = distances[:, 1:].flatten()
    dc = np.percentile(all_dist, dc_percent)

    rho = np.sum(np.exp(-(distances / dc) ** 2), axis=1)

    order = np.argsort(rho)[::-1]
    delta = np.zeros(n)
    nearest_higher = np.zeros(n, dtype=int)
    delta[order[0]] = np.max(distances[order[0]])
    for i in range(1, n):
        idx = order[i]
        higher = order[:i]
        d_to_higher = distances[idx, higher]
        min_idx = np.argmin(d_to_higher)
        delta[idx] = d_to_higher[min_idx]
        nearest_higher[idx] = higher[min_idx]

    gamma = rho * delta
    return rho, delta, gamma, order, nearest_higher


"""
根据中心点和最近更高密度点分配簇标签。

Parameters
----------
n : int
    样本数。
centers : ndarray
    中心点索引。
nearest_higher : ndarray
    最近更高密度点索引。
order : ndarray
    按 rho 降序排列的索引。

Returns
-------
labels : ndarray
    每个样本的簇标签（-1 表示未分配）。
"""
def assign_clusters(n, centers, nearest_higher, order):
    labels = -np.ones(n, dtype=int)
    for i, c in enumerate(centers):
        labels[c] = i
    for idx in order:
        if labels[idx] == -1:
            labels[idx] = labels[nearest_higher[idx]]
    return labels


"""
DPC 完整流水线：训练 → 决策图 → 聚类 → 评估 → PCA 可视化。

Parameters
----------
X : DataFrame
    全部特征（已标准化，去标签）。
y_true : array-like, optional
    真实标签（若提供则计算 ARI）。
n_clusters : int, optional
    聚类数，默认 3。
dc_percent : float, optional
    截断距离百分比，默认 2.0。

Returns
-------
centers : ndarray
    聚类中心索引。
y_pred : ndarray
    聚类标签。
"""
def run_dpc_pipeline(X, y_true=None, n_clusters=3, dc_percent=2.0):
    X_arr = X.values if hasattr(X, 'values') else X
    rho, delta, gamma, order, nearest_higher = density_peaks(X_arr, dc_percent)
    centers = order[:n_clusters]
    y_pred = assign_clusters(X_arr.shape[0], centers, nearest_higher, order)

    # 评估
    sil = silhouette_score(X_arr, y_pred)
    print(f'\n{"="*60}')
    print(f'  密度峰值聚类 (DPC, n_clusters={n_clusters}, dc%={dc_percent})')
    print(f'{"="*60}')
    print(f'  轮廓系数: {sil:.4f}')
    if y_true is not None:
        ari = adjusted_rand_score(y_true, y_pred)
        print(f'  ARI: {ari:.4f}')

    # 决策图
    plt.figure(figsize=(8,6))
    plt.scatter(rho, delta, c='steelblue', s=50, alpha=0.7)
    plt.scatter(rho[centers], delta[centers], c='red', s=100, marker='X', label='聚类中心')
    plt.xlabel('局部密度 ρ')
    plt.ylabel('距离 δ')
    plt.title('DPC 决策图 (ρ vs δ)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # PCA 可视化
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_arr)
    plt.figure(figsize=(10,6))
    plt.scatter(X_pca[:,0], X_pca[:,1], c=y_pred, cmap='Set1', edgecolors='k', s=60)
    plt.scatter(X_pca[centers,0], X_pca[centers,1], c='black', marker='X', s=200, label='中心')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    plt.title(f'DPC 聚类结果 | 轮廓系数: {sil:.3f}')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return centers, y_pred
