"""KNN 分类器 — 训练、评估与可视化"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA

"""
训练 KNN 并打印评估结果。

Parameters
----------
X_train, X_test : DataFrame
    训练/测试特征
y_train, y_test : array-like
    训练/测试标签（已编码为整数）
class_names : array-like
    原始类别名称，用于报告显示
n_neighbors : int
    K 近邻数

Returns
-------
model : KNeighborsClassifier
    训练好的模型
y_pred : ndarray
    测试集预测结果
"""
def train_and_evaluate(X_train, X_test, y_train, y_test, class_names, n_neighbors=3):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f'\n{"="*50}')
    print(f'  KNN 分类报告 (k={n_neighbors}, 全部 {X_train.shape[1]} 个特征)')
    print(f'{"="*50}')
    print(classification_report(y_test, y_pred, target_names=class_names))

    return model, y_pred

"""绘制混淆矩阵热力图"""
def plot_confusion_matrix(y_test, y_pred, class_names, accuracy):

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f'混淆矩阵 — KNN (全部特征)\n准确率: {accuracy:.2%}')
    plt.tight_layout()
    plt.show()


"""
PCA 降维到 2D，绘制 KNN 决策边界。

把全部特征投影到 2D 主成分空间，在这个空间训练 KNN 并画出分类区域。
"""
def plot_pca_decision_boundary(X, y, class_names, n_neighbors=3):

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
    plt.title(f'PCA 降维 + KNN 决策边界 (k={n_neighbors}, 全部特征)\n'
              f'累计解释方差: {pca.explained_variance_ratio_.sum():.1%}  |  '
              f'准确率: {model.score(X_test, y_test):.2%}')
    plt.colorbar(scatter, ticks=[0, 1, 2], label='类别')
    plt.tight_layout()
    plt.show()


"""
特征对散点图矩阵。

横纵坐标覆盖全部 4 个特征的两两组合，按真实类别着色。
另外打印预测错误的样本。
"""
def plot_pairplot(X_test, y_test, y_pred, le):

    # 组合预测结果
    df_viz = X_test.copy()
    df_viz['真实类别'] = le.inverse_transform(y_test)
    df_viz['预测类别'] = le.inverse_transform(y_pred)
    df_viz['结果'] = df_viz.apply(
        lambda row: '✓ 正确' if row['真实类别'] == row['预测类别'] else '✗ 错误',
        axis=1
    )

    # Pairplot：只画特征 + 真实类别
    df_plot = X_test.copy()
    df_plot['类别'] = le.inverse_transform(y_test)

    g = sns.pairplot(df_plot, hue='类别', palette='Set1', diag_kind='hist',
                     plot_kws={'alpha': 0.7, 's': 50})
    g.fig.suptitle('全部特征 — 测试集散点图矩阵 (按真实类别着色)', y=1.02, fontsize=14)
    plt.show()

    # 打印错误样本
    n_errors = sum(y_test != y_pred)
    if n_errors > 0:
        print(f'\n=== 预测错误的样本 ({n_errors} 个) ===')
        print(df_viz[df_viz['结果'] == '✗ 错误'][df_viz.columns[:-1]].to_string())
    else:
        print('\n🎉 全部预测正确，没有错误样本！')

"""
KNN 完整流水线：训练 → 评估 → 三种可视化。

这是供 main.py 调用的统一入口。
"""
def run_knn_pipeline(X_train, X_test, y_train, y_test, X_full, y_encoded, le, n_neighbors=3):
    # 1. 训练 + 评估
    model, y_pred = train_and_evaluate(
        X_train, X_test, y_train, y_test, le.classes_, n_neighbors
    )
    accuracy = model.score(X_test, y_test)

    # 2. 混淆矩阵
    plot_confusion_matrix(y_test, y_pred, le.classes_, accuracy)

    # 3. PCA 降维决策边界
    plot_pca_decision_boundary(X_full, y_encoded, le.classes_, n_neighbors)

    # 4. Pairplot 特征矩阵 + 错误分析
    plot_pairplot(X_test, y_test, y_pred, le)

    return model, y_pred
