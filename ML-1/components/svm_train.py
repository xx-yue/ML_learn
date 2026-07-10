"""SVM 分类器 — 训练、评估与可视化"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA

"""训练 SVM 分类器并打印分类报告。

Parameters
----------
X_train, X_test : DataFrame
    训练/测试特征
y_train, y_test : array-like
    训练/测试标签
class_names : array-like
    原始类别名称
**kwargs : dict
    传递给 SVC 的参数（如 kernel、C、gamma 等）

Returns
-------
model : SVC
    训练好的模型
y_pred : ndarray
    测试集预测结果"""
def train_and_evaluate(X_train, X_test, y_train, y_test, class_names, **kwargs):
    model = SVC(**kwargs)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f'\n{"="*50}')
    print(f'  SVM 分类报告 (kernel={kwargs.get("kernel","rbf")})')
    print(f'{"="*50}')
    print(classification_report(y_test, y_pred, target_names=class_names))
    return model, y_pred

"""绘制混淆矩阵热力图，展示真实类别 vs 预测类别的分布。"""
def plot_confusion_matrix(y_test, y_pred, class_names, accuracy):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Greens', values_format='d')
    plt.title(f'混淆矩阵 — SVM\n准确率: {accuracy:.2%}')
    plt.tight_layout()
    plt.show()

"""PCA 降维到 2D，在降维后空间训练 SVM 并绘制决策边界。"""
def plot_pca_decision_boundary(X, y, class_names, **kwargs):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.2, random_state=42, stratify=y)
    model = SVC(**kwargs)
    model.fit(X_train, y_train)

    x_min, x_max = X_pca[:, 0].min()-1, X_pca[:, 0].max()+1
    y_min, y_max = X_pca[:, 1].min()-1, X_pca[:, 1].max()+1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.figure(figsize=(10,6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='Set1')
    plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap='Set1', edgecolors='k', s=60)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    plt.title(f'SVM 决策边界 (kernel={kwargs.get("kernel","rbf")})\n准确率: {model.score(X_test,y_test):.2%}')
    plt.colorbar(ticks=[0,1,2], label='类别')
    plt.tight_layout()
    plt.show()

"""SVM 完整流水线：训练 → 评估 → 混淆矩阵 → PCA 决策边界。供 main.py 调用的统一入口。"""
def run_svm_pipeline(X_train, X_test, y_train, y_test, X_full, y_encoded, le, **kwargs):
    model, y_pred = train_and_evaluate(X_train, X_test, y_train, y_test, le.classes_, **kwargs)
    accuracy = model.score(X_test, y_test)
    plot_confusion_matrix(y_test, y_pred, le.classes_, accuracy)
    plot_pca_decision_boundary(X_full, y_encoded, le.classes_, **kwargs)
    return model, y_pred