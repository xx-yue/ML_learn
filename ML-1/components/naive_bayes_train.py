"""朴素贝叶斯分类器 — 训练、评估与可视化"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA


def train_and_evaluate(X_train, X_test, y_train, y_test, class_names):
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f'\n{"="*50}')
    print('  朴素贝叶斯分类报告')
    print(f'{"="*50}')
    print(classification_report(y_test, y_pred, target_names=class_names))
    return model, y_pred


def plot_confusion_matrix(y_test, y_pred, class_names, accuracy):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Purples', values_format='d')
    plt.title(f'混淆矩阵 — 朴素贝叶斯\n准确率: {accuracy:.2%}')
    plt.tight_layout()
    plt.show()


def plot_pca_decision_boundary(X, y, class_names):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.2, random_state=42, stratify=y)
    model = GaussianNB()
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
    plt.title(f'朴素贝叶斯决策边界\n准确率: {model.score(X_test, y_test):.2%}')
    plt.colorbar(ticks=[0, 1, 2], label='类别')
    plt.tight_layout()
    plt.show()


def run_nb_pipeline(X_train, X_test, y_train, y_test, X_full, y_encoded, le):
    model, y_pred = train_and_evaluate(X_train, X_test, y_train, y_test, le.classes_)
    accuracy = model.score(X_test, y_test)
    plot_confusion_matrix(y_test, y_pred, le.classes_, accuracy)
    plot_pca_decision_boundary(X_full, y_encoded, le.classes_)
    return model, y_pred