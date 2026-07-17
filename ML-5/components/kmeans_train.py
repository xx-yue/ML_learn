"""K-means 聚类 — Wine Quality 数据集（去标签）训练、评估与可视化"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score, adjusted_mutual_info_score


"""
K-means 聚类完整流水线：训练 → 评估 → 可视化。

Parameters
----------
X : DataFrame
    全部特征（已标准化，去标签）。
y_true : array-like, optional
    真实标签（若提供则计算外部指标 ARI/AMI）。
n_clusters : int, optional
    聚类数，默认 6。
random_state : int, optional
    随机种子，默认 42。

Returns
-------
model : KMeans
    训练好的模型。
y_pred : ndarray
    聚类标签。
"""
def run_kmeans_pipeline(X, y_true=None, n_clusters=6, random_state=42):
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    y_pred = model.fit_predict(X)

    # 内部评估：轮廓系数
    sil = silhouette_score(X, y_pred)
    print(f'\n{"="*60}')
    print(f'  K-means 聚类 (k={n_clusters})')
    print(f'{"="*60}')
    print(f'  轮廓系数 (Silhouette): {sil:.4f}')
    if y_true is not None:
        ari = adjusted_rand_score(y_true, y_pred)
        ami = adjusted_mutual_info_score(y_true, y_pred)
        print(f'  调整兰德指数 (ARI):  {ari:.4f}')
        print(f'  调整互信息 (AMI):   {ami:.4f}')

    # PCA 降维可视化
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(10,6))
    scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=y_pred, cmap='Set1', edgecolors='k', s=60)
    centers_pca = pca.transform(model.cluster_centers_)
    plt.scatter(centers_pca[:,0], centers_pca[:,1], c='black', marker='X', s=200, label='聚类中心')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    title_str = f'K-means 聚类 (k={n_clusters}) | 轮廓系数: {sil:.3f}'
    if y_true is not None:
        title_str += f' | ARI: {ari:.3f}'
    plt.title(title_str)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 肘部法则图（可选）
    inertias = []
    K_range = range(2, 11)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)
    plt.figure(figsize=(8,5))
    plt.plot(K_range, inertias, 'bo-')
    plt.xlabel('K')
    plt.ylabel('惯性 (Inertia)')
    plt.title('肘部法则 — 选择最佳 K')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return model, y_pred
