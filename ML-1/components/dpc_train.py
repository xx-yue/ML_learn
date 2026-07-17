"""密度峰值聚类 (DPC, Rodriguez & Laio 2014) — 手写简化版"""

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
    特征矩阵 (n_samples, n_features)
dc_percent : float
    截断距离百分比，默认 2%

Returns
-------
order : ndarray
    按 rho 降序排列的索引
rho : ndarray
    局部密度
delta : ndarray
    到更高密度点的最小距离
gamma : ndarray
    rho * delta
nearest_higher : ndarray
    每个点的最近更高密度点索引
"""
def density_peaks(X, dc_percent=2.0):
    n = X.shape[0]
    nbrs = NearestNeighbors(n_neighbors=n).fit(X)
    distances, _ = nbrs.kneighbors(X)

    # 截断距离 dc
    all_dist = distances[:, 1:].flatten()
    dc = np.percentile(all_dist, dc_percent)

    # 局部密度 rho（高斯核）
    rho = np.sum(np.exp(-(distances / dc) ** 2), axis=1)

    # delta：到更高密度点的最小距离
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
    return order, rho, delta, gamma, nearest_higher


"""
根据中心点和最近更高密度点分配簇标签。

把非中心样本分配到"最近更高密度点所属簇"。
按密度降序遍历：每个未分配样本继承其 nearest_higher 的标签。
"""
def assign_clusters(X, centers_idx, nearest_higher, rho):
    n = X.shape[0]
    labels = -np.ones(n, dtype=int)
    for i, c in enumerate(centers_idx):
        labels[c] = i
    # 按密度降序分配
    order = np.argsort(rho)[::-1]
    for idx in order:
        if labels[idx] == -1:
            labels[idx] = labels[nearest_higher[idx]]
    return labels


"""
DPC 完整流水线：训练 → 决策图 → 聚类 → 评估 → PCA 可视化。

Parameters
----------
X : DataFrame
    全部特征（已标准化，去标签）
y_true : array-like
    真实标签（用于计算 ARI 外部指标）
n_clusters : int
    聚类数，默认 3
dc_percent : float
    截断距离百分比，默认 1.0

Returns
-------
centers : ndarray
    聚类中心索引
y_pred : ndarray
    聚类标签
"""
def run_dpc_pipeline(X, y_true, n_clusters=3, dc_percent=1.0, random_state=42):
    order, rho, delta, gamma, nearest_higher = density_peaks(X, dc_percent)

    # 选前 n_clusters 个 gamma 最大的作为中心
    centers = order[:n_clusters]

    y_pred = assign_clusters(X, centers, nearest_higher, rho)

    # 评估
    sil = silhouette_score(X, y_pred)
    print(f'\n{"="*50}')
    print(f'  密度峰值聚类 (DPC, n_clusters={n_clusters}, dc%={dc_percent})')
    print(f'{"="*50}')
    print(f'  轮廓系数: {sil:.4f}')
    if y_true is not None:
        ari = adjusted_rand_score(y_true, y_pred)
        print(f'  ARI: {ari:.4f}')

    # 决策图 (rho vs delta，DPC 经典图)
    plt.figure(figsize=(8, 6))
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
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(10, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap='Set1', edgecolors='k', s=60)
    plt.scatter(X_pca[centers, 0], X_pca[centers, 1], c='black', marker='X', s=200, label='中心')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    plt.title(f'DPC 聚类结果 | 轮廓系数: {sil:.3f}')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return centers, y_pred