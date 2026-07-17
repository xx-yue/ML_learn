"""KNN 分类器 — Glass 数据集（9 维特征）训练、评估与可视化"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA

"""训练 KNN 并打印评估结果。

Parameters
----------
X_train, X_test : DataFrame
    训练/测试特征（已标准化）
y_train, y_test : array-like
    训练/测试标签（已编码为 0-5）
class_names : array-like
    原始类别名称
n_neighbors : int
    K 近邻数（默认 5）

Returns
-------
model : KNeighborsClassifier
    训练好的模型
y_pred : ndarray
    测试集预测结果"""
def train_and_evaluate(X_train, X_test, y_train, y_test, class_names, n_neighbors=5):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 确保类别名称为字符串
    class_names = [str(c) for c in class_names]

    print(f'\n{"="*60}')
    print(f'  KNN 分类报告 (k={n_neighbors}, 全部 {X_train.shape[1]} 个特征)')
    print(f'{"="*60}')
    print(classification_report(y_test, y_pred, target_names=class_names))

    return model, y_pred


"""绘制混淆矩阵热力图，展示真实类别 vs 预测类别的分布。"""
def plot_confusion_matrix(y_test, y_pred, class_names, accuracy):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f'混淆矩阵 — KNN (全部 9 个特征, k=5)\n准确率: {accuracy:.2%}')
    plt.tight_layout()
    plt.show()


"""PCA 降维到 2D，绘制 KNN 决策边界。

Glass 有 9 个特征，无法直接可视化。通过 PCA 压缩到 2 维后，
在低维空间重新训练 KNN 并画出分类区域。
"""
def plot_pca_decision_boundary(X, y, class_names, n_neighbors=5):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # 在 PCA 空间划分并训练
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.2, random_state=42, stratify=y
    )
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)

    # 生成网格
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.1),
        np.arange(y_min, y_max, 0.1)
    )
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='Set1')
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y,
                          cmap='Set1', edgecolors='k', s=60)
    plt.xlabel(f'主成分1 (解释方差: {pca.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'主成分2 (解释方差: {pca.explained_variance_ratio_[1]:.1%})')
    plt.title(f'PCA 降维 + KNN 决策边界 (k={n_neighbors}, 全部 9 个特征)\n'
              f'累计解释方差: {pca.explained_variance_ratio_.sum():.1%}  |  '
              f'准确率: {model.score(X_test, y_test):.2%}')
    plt.colorbar(scatter, ticks=[0, 1, 2, 3, 4, 5], label='类别')
    plt.tight_layout()
    plt.show()


"""特征相关性热力图。

Glass 有 9 个特征，用相关性热力图展示特征之间的线性关系强度。
"""
def plot_correlation_heatmap(X, y_encoded, class_names):
    # 组合特征 + 标签
    df_corr = X.copy()
    df_corr['类别'] = y_encoded

    # 按类别分组计算各特征均值
    print(f'\n{"="*60}')
    print('  各类别特征均值对比')
    print(f'{"="*60}')
    for i, name in enumerate(class_names):
        print(f'\n--- {name} ---')
        print(df_corr[df_corr['类别'] == i].drop('类别', axis=1).mean().to_string())

    # 绘制特征相关性矩阵
    plt.figure(figsize=(12, 10))
    corr = X.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)  # 只显示下三角
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
                cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5,
                cbar_kws={'shrink': 0.8, 'label': '相关系数'})
    plt.title('Glass 数据集 9 个特征相关性矩阵', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


"""Top-2 最有区分力特征的散点图 + 类别分布。

用 PCA 方差解释率选出最重要的 2 个特征组合，
展示样本在最有区分力的二维空间中的分布。
"""
def plot_top_features_scatter(X, y_encoded, le, n_neighbors=5):
    # 使用 PCA 找出区分力最强的 2 个主成分
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # 同时找出与第 1、2 主成分最相关的原始特征
    comp1 = pd.Series(pca.components_[0], index=X.columns)
    comp2 = pd.Series(pca.components_[1], index=X.columns)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 子图1: PCA 空间散点图
    for i, name in enumerate(le.classes_):
        mask = y_encoded == i
        axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                        label=name, alpha=0.7, edgecolors='k', s=50)
    axes[0].set_xlabel(f'主成分 1 ({pca.explained_variance_ratio_[0]:.1%})')
    axes[0].set_ylabel(f'主成分 2 ({pca.explained_variance_ratio_[1]:.1%})')
    axes[0].set_title(f'PCA 降维散点图\n(累计解释方差: {pca.explained_variance_ratio_.sum():.1%})')
    axes[0].legend()

    # 子图2: 主成分载荷（各原始特征对前 2 个主成分的贡献）
    loadings = pd.DataFrame({
        'PC1': pca.components_[0],
        'PC2': pca.components_[1]
    }, index=X.columns)
    axes[1].barh(range(len(X.columns)), np.abs(loadings).sum(axis=1))
    axes[1].set_yticks(range(len(X.columns)))
    axes[1].set_yticklabels(X.columns)
    axes[1].set_xlabel('对主成分的总贡献度')
    axes[1].set_title('各特征对 PCA 前 2 主成分的总贡献')

    plt.suptitle('Glass 数据集 — PCA 降维可视化 + 特征贡献分析', fontsize=14)
    plt.tight_layout()
    plt.show()

    # 重新训练一次用于错误分析
    from sklearn.model_selection import cross_val_predict
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X.values, y_encoded)  # 用全量数据训练（只用于错误分析）
    y_pred = cross_val_predict(model, X.values, y_encoded, cv=5)

    n_errors = sum(y_encoded != y_pred)
    if n_errors > 0:
        print(f'\n5 折交叉验证错误样本: {n_errors}/{len(y_encoded)} = {n_errors/len(y_encoded):.1%}')
    else:
        print('\n5 折交叉验证全部正确！')


"""KNN 完整流水线：训练 → 评估 → 四种可视化。

Glass 专用流水线，相比 Iris 做了以下调整：
- 用相关性热力图代替 pairplot（9 维特征 pairplot 过于庞大）
- 增加 PCA 特征贡献分析，展示 9 个特征中哪些最有区分力

供 main.py 调用的统一入口。"""
def run_knn_pipeline(X_train, X_test, y_train, y_test, X_full, y_encoded, le, n_neighbors=5):
    # 1. 训练 + 评估
    model, y_pred = train_and_evaluate(
        X_train, X_test, y_train, y_test, le.classes_, n_neighbors
    )
    accuracy = model.score(X_test, y_test)

    # 2. 混淆矩阵
    plot_confusion_matrix(y_test, y_pred, le.classes_, accuracy)

    # 3. PCA 降维决策边界
    plot_pca_decision_boundary(X_full, y_encoded, le.classes_, n_neighbors)

    # 4. 特征相关性热力图 + 各类别均值
    plot_correlation_heatmap(X_full, y_encoded, le.classes_)

    # 5. PCA 散点 + 特征贡献分析
    plot_top_features_scatter(X_full, y_encoded, le, n_neighbors)

    return model, y_pred
