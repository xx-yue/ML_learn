"""K-means 聚类 — 训练、评估与可视化（Iris 去标签）"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score, adjusted_mutual_info_score


def run_kmeans_pipeline(X, y_true, n_clusters=3, random_state=42):
    """
    X: 全部特征（去标签）
    y_true: 原始标签（用于算 ARI/AMI 外部指标，可选）
    """
    # 1. K-means 训练
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    y_pred = model.fit_predict(X)

    # 2. 评估
    sil = silhouette_score(X, y_pred)
    print(f'\n{"="*50}')
    print(f'  K-means 聚类 (k={n_clusters})')
    print(f'{"="*50}')
    print(f'  轮廓系数 (Silhouette): {sil:.4f}')
    if y_true is not None:
        ari = adjusted_rand_score(y_true, y_pred)
        ami = adjusted_mutual_info_score(y_true, y_pred)
        print(f'  调整兰德指数 (ARI):  {ari:.4f}')
        print(f'  调整互信息 (AMI):   {ami:.4f}')

    # 3. PCA 降维可视化聚类结果
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred,
                          cmap='Set1', edgecolors='k', s=60)
    # 标中心点
    centers_pca = pca.transform(model.cluster_centers_)
    plt.scatter(centers_pca[:, 0], centers_pca[:, 1],
                c='black', marker='X', s=200, label='聚类中心')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    title_str = f'K-means 聚类 (k={n_clusters}) | 轮廓系数: {sil:.3f}'
    if y_true is not None:
        title_str += f' | ARI: {ari:.3f}'
    plt.title(title_str)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return model, y_pred