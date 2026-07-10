"""决策树分类器 — 训练、评估与可视化"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA

"""
训练决策树并打印评估结果。

Returns
-------
model : DecisionTreeClassifier
y_pred : ndarray
"""
def train_and_evaluate(X_train, X_test, y_train, y_test, class_names,
                       criterion='gini', max_depth=None, random_state=42):
    model = DecisionTreeClassifier(
        criterion=criterion, max_depth=max_depth, random_state=random_state
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    depth_info = f'未限制' if max_depth is None else max_depth
    print(f'\n{"="*50}')
    print(f'  决策树分类报告 (criterion={criterion}, max_depth={depth_info})')
    print(f'{"="*50}')
    print(classification_report(y_test, y_pred, target_names=class_names))

    return model, y_pred

"""绘制决策树结构图 — 决策树独有的可视化"""
def plot_tree_structure(model, feature_names, class_names):
    plt.figure(figsize=(16, 10))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=list(class_names),
        filled=True,           # 用颜色区分类别
        rounded=True,          # 圆角节点
        fontsize=10,
        impurity=True,         # 显示基尼系数/熵
        proportion=True        # 显示比例而非绝对数量
    )
    criterion = model.criterion
    plt.title(f'决策树结构 ({criterion}, max_depth={model.max_depth})', fontsize=14)
    plt.tight_layout()
    plt.show()

"""混淆矩阵热力图"""
def plot_confusion_matrix(y_test, y_pred, class_names, accuracy):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f'混淆矩阵 — 决策树\n准确率: {accuracy:.2%}')
    plt.tight_layout()
    plt.show()




"""
PCA 降维到 2D，绘制决策树决策边界。
"""
def plot_pca_decision_boundary(X, y, model, class_names):

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    from sklearn.model_selection import train_test_split

    # 在 PCA 空间训练同样的决策树
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.2, random_state=42, stratify=y
    )
    dt_pca = DecisionTreeClassifier(
        criterion=model.criterion, max_depth=model.max_depth,
        random_state=model.random_state
    )
    dt_pca.fit(X_train, y_train)

    # 网格
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.02),
        np.arange(y_min, y_max, 0.02)
    )
    Z = dt_pca.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='Set1')
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y,
                          cmap='Set1', edgecolors='k', s=60)
    plt.xlabel(f'主成分1 (解释方差: {pca.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'主成分2 (解释方差: {pca.explained_variance_ratio_[1]:.1%})')
    plt.title(f'PCA 降维 + 决策树决策边界 (全部特征)\n'
              f'累计解释方差: {pca.explained_variance_ratio_.sum():.1%}  |  '
              f'准确率: {dt_pca.score(X_test, y_test):.2%}')
    plt.colorbar(scatter, ticks=[0, 1, 2], label='类别')
    plt.tight_layout()
    plt.show()



"""特征重要性柱状图 — 决策树独有的分析"""
def plot_feature_importance(model, feature_names):

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(range(len(importances)), importances[indices],
                   color=['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3'])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices],
               rotation=30, ha='right')
    plt.ylabel('重要性')
    plt.title('特征重要性 (决策树)')
    # 在柱子上方标数值
    for bar, val in zip(bars, importances[indices]):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f'{val:.3f}', ha='center', fontsize=10)
    plt.tight_layout()
    plt.show()


"""测试集特征对散点图矩阵 + 错误样本打印"""
def plot_pairplot(X_test, y_test, y_pred, le):

    df_viz = X_test.copy()
    df_viz['真实类别'] = le.inverse_transform(y_test)
    df_viz['预测类别'] = le.inverse_transform(y_pred)
    df_viz['结果'] = df_viz.apply(
        lambda row: '✓ 正确' if row['真实类别'] == row['预测类别'] else '✗ 错误',
        axis=1
    )

    df_plot = X_test.copy()
    df_plot['类别'] = le.inverse_transform(y_test)

    g = sns.pairplot(df_plot, hue='类别', palette='Set1', diag_kind='hist',
                     plot_kws={'alpha': 0.7, 's': 50})
    g.fig.suptitle('全部特征 — 测试集散点图矩阵 (决策树, 按真实类别着色)',
                   y=1.02, fontsize=14)
    plt.show()

    n_errors = sum(y_test != y_pred)
    if n_errors > 0:
        print(f'\n=== 预测错误的样本 ({n_errors} 个) ===')
        print(df_viz[df_viz['结果'] == '✗ 错误'][df_viz.columns[:-1]].to_string())
    else:
        print('\n所有样本预测正确！')


"""
决策树完整流水线：训练 → 评估 → 四种可视化。

Parameters
----------
criterion : 'gini' | 'entropy'
    分裂准则：基尼系数 / 信息增益
max_depth : int | None
    最大深度，None 表示不限制（可能过拟合）
"""
def run_dt_pipeline(X_train, X_test, y_train, y_test, X_full, y_encoded, le,
                    criterion='gini', max_depth=None):

    feature_names = X_train.columns.tolist()

    # 1. 训练 + 评估
    model, y_pred = train_and_evaluate(
        X_train, X_test, y_train, y_test, le.classes_,
        criterion=criterion, max_depth=max_depth
    )
    accuracy = model.score(X_test, y_test)

    # 2. 决策树结构图（决策树独有）
    plot_tree_structure(model, feature_names, le.classes_)

    # 3. 特征重要性（决策树独有）
    plot_feature_importance(model, feature_names)

    # 4. 混淆矩阵
    plot_confusion_matrix(y_test, y_pred, le.classes_, accuracy)

    # 5. PCA 降维决策边界
    plot_pca_decision_boundary(X_full, y_encoded, model, le.classes_)

    # 6. Pairplot
    plot_pairplot(X_test, y_test, y_pred, le)

    return model, y_pred
