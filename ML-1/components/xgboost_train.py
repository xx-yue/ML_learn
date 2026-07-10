"""XGBoost 分类器 — 训练、评估与可视化（CPU 模式）"""

import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA


"""训练 XGBoost 分类器并打印分类报告。

Parameters
----------
**kwargs : dict
    传递给 XGBClassifier 的参数（如 n_estimators、max_depth、tree_method 等）

Returns
-------
model : XGBClassifier
    训练好的模型
y_pred : ndarray
    测试集预测结果"""
def train_and_evaluate(X_train, X_test, y_train, y_test, class_names, **kwargs):
    model = XGBClassifier(**kwargs)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f'\n{"="*50}')
    print('  XGBoost 分类报告')
    print(f'{"="*50}')
    print(classification_report(y_test, y_pred, target_names=class_names))
    return model, y_pred


"""绘制混淆矩阵热力图，展示真实类别 vs 预测类别的分布。"""
def plot_confusion_matrix(y_test, y_pred, class_names, accuracy):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='RdPu', values_format='d')
    plt.title(f'混淆矩阵 — XGBoost\n准确率: {accuracy:.2%}')
    plt.tight_layout()
    plt.show()


"""绘制特征重要性柱状图。XGBoost 基于梯度提升过程中特征被用于分裂的次数和增益综合计算贡献度。"""
def plot_feature_importance(model, feature_names):
    # XGBoost 自带特征重要性图，这里用条形图保持风格一致
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(range(len(importances)), importances[indices],
                   color=['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c'])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices],
               rotation=30, ha='right')
    plt.ylabel('重要性')
    plt.title('特征重要性 (XGBoost)')
    for bar, val in zip(bars, importances[indices]):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f'{val:.3f}', ha='center', fontsize=10)
    plt.tight_layout()
    plt.show()


"""PCA 降维到 2D，在降维后空间训练 XGBoost 并绘制决策边界。"""
def plot_pca_decision_boundary(X, y, class_names, **kwargs):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.2, random_state=42, stratify=y)
    model = XGBClassifier(**kwargs)
    model.fit(X_train, y_train)

    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='Set1')
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='Set1', edgecolors='k', s=60)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    plt.title(f'XGBoost 决策边界\n准确率: {model.score(X_test, y_test):.2%}')
    plt.colorbar(ticks=[0, 1, 2], label='类别')
    plt.tight_layout()
    plt.show()


"""XGBoost 完整流水线：训练 → 评估 → 混淆矩阵 → 特征重要性 → PCA 决策边界。

默认强制 CPU 模式（tree_method='hist'），评估指标使用 mlogloss。
供 main.py 调用的统一入口。"""
def run_xgb_pipeline(X_train, X_test, y_train, y_test, X_full, y_encoded, le, **kwargs):
    # 强制使用 CPU 模式，因为你的环境不支持 GPU
    if 'tree_method' not in kwargs:
        kwargs['tree_method'] = 'hist'
    if 'eval_metric' not in kwargs:
        kwargs['eval_metric'] = 'mlogloss'
    if 'random_state' not in kwargs:
        kwargs['random_state'] = 42
    if 'use_label_encoder' not in kwargs:
        kwargs['use_label_encoder'] = False  # 避免警告
    model, y_pred = train_and_evaluate(X_train, X_test, y_train, y_test, le.classes_, **kwargs)
    accuracy = model.score(X_test, y_test)
    plot_confusion_matrix(y_test, y_pred, le.classes_, accuracy)
    plot_feature_importance(model, X_train.columns.tolist())
    plot_pca_decision_boundary(X_full, y_encoded, le.classes_, **kwargs)
    return model, y_pred