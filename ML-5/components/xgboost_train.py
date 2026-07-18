"""XGBoost 分类器 — Wine Quality 数据集（11 维特征）训练、评估与可视化（CPU 模式）"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score, roc_auc_score, accuracy_score
from sklearn.decomposition import PCA


"""
训练 XGBoost 并打印分类报告。

Parameters
----------
X_train, X_test : DataFrame
    训练/测试特征（已标准化）。
y_train, y_test : array-like
    训练/测试标签（已编码）。
class_names : array-like
    原始类别名称。
**kwargs : dict
    传递给 XGBClassifier 的参数。

Returns
-------
model : XGBClassifier
    训练好的模型。
y_pred : ndarray
    测试集预测结果。
"""
def train_and_evaluate(X_train, X_test, y_train, y_test, class_names, **kwargs):
    model = XGBClassifier(**kwargs)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    class_names = [str(c) for c in class_names]
    print(f'\n{"="*60}')
    print(f'  XGBoost 分类报告 (n_estimators={kwargs.get("n_estimators",100)})')
    print(f'{"="*60}')
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
    # 综合评估指标：准确率、F1值、AUC
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    print(f'\n  准确率 (Accuracy):    {acc:.4f}')
    print(f'  F1值 (Macro):         {f1_macro:.4f}')
    print(f'  F1值 (Weighted):      {f1_weighted:.4f}')
    try:
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
            auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
            print(f'  AUC (OvR Macro):      {auc:.4f}')
        elif hasattr(model, 'decision_function'):
            y_score = model.decision_function(X_test)
            auc = roc_auc_score(y_test, y_score, multi_class='ovr', average='macro')
            print(f'  AUC (OvR Macro):      {auc:.4f}')
        else:
            print(f'  AUC:                  不支持（模型无 predict_proba / decision_function）')
    except Exception as e:
        print(f'  AUC:                  无法计算 ({e})')

    return model, y_pred


"""
绘制混淆矩阵热力图。
"""
def plot_confusion_matrix(y_test, y_pred, class_names, accuracy):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='RdPu', values_format='d')
    plt.title(f'混淆矩阵 — XGBoost\n准确率: {accuracy:.2%}')
    plt.tight_layout(); plt.show()


"""
绘制 XGBoost 特征重要性条形图。
"""
def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10,6))
    bars = plt.bar(range(len(importances)), importances[indices],
                   color=['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99',
                          '#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a',
                          '#ffff99'][:len(importances)])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices],
               rotation=45, ha='right')
    plt.ylabel('重要性')
    plt.title('特征重要性 (XGBoost)')
    for bar, val in zip(bars, importances[indices]):
        plt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                 f'{val:.3f}', ha='center', fontsize=9)
    plt.tight_layout(); plt.show()


"""
PCA 降维到 2D，绘制 XGBoost 决策边界。
"""
def plot_pca_decision_boundary(X, y, class_names, **kwargs):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.2, random_state=42, stratify=y)
    model = XGBClassifier(**kwargs)
    model.fit(X_train, y_train)
    x_min, x_max = X_pca[:,0].min()-1, X_pca[:,0].max()+1
    y_min, y_max = X_pca[:,1].min()-1, X_pca[:,1].max()+1
    xx, yy = np.meshgrid(np.arange(x_min,x_max,0.1), np.arange(y_min,y_max,0.1))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.figure(figsize=(10,6))
    plt.contourf(xx,yy,Z,alpha=0.3,cmap='Set1')
    scatter = plt.scatter(X_pca[:,0],X_pca[:,1],c=y,cmap='Set1',edgecolors='k',s=60)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    plt.title(f'XGBoost 决策边界\n准确率: {model.score(X_test,y_test):.2%}')
    plt.colorbar(scatter, ticks=[0, 1, 2, 3, 4, 5], label='类别')
    plt.tight_layout(); plt.show()


"""
绘制特征相关性热力图。
"""
def plot_correlation_heatmap(X, y_encoded, class_names):
    df_corr = X.copy()
    df_corr['类别'] = y_encoded
    print(f'\n{"="*60}\n  各类别特征均值对比\n{"="*60}')
    for i, name in enumerate(class_names):
        print(f'\n--- {name} ---')
        print(df_corr[df_corr['类别']==i].drop('类别',axis=1).mean().to_string())
    plt.figure(figsize=(14, 12))
    corr = X.corr()
    mask = np.triu(np.ones_like(corr,dtype=bool),k=1)
    sns.heatmap(corr,mask=mask,annot=True,fmt='.2f',cmap='RdBu_r',
                center=0,vmin=-1,vmax=1,square=True,linewidths=0.5,
                cbar_kws={'shrink':0.8,'label':'相关系数'})
    plt.title('Wine Quality 数据集 11 个特征相关性矩阵',fontsize=14)
    plt.xticks(rotation=45,ha='right'); plt.yticks(rotation=0)
    plt.tight_layout(); plt.show()


"""
PCA 散点图 + 特征贡献分析。
"""
def plot_top_features_scatter(X, y_encoded, le, **kwargs):
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)

    fig = plt.figure(figsize=(18, 6))

    # 子图1: PCA 3D 空间散点图（区分力最强的 3 个主成分）
    ax1 = fig.add_subplot(121, projection='3d')
    for i, name in enumerate(le.classes_):
        mask = y_encoded == i
        ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2],
                    label=name, alpha=0.7, edgecolors='k', s=50)
    ax1.set_xlabel(f'主成分 1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax1.set_ylabel(f'主成分 2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax1.set_zlabel(f'主成分 3 ({pca.explained_variance_ratio_[2]:.1%})')
    ax1.set_title(f'PCA 3D 降维散点图\n(累计解释方差: {pca.explained_variance_ratio_.sum():.1%})')
    ax1.legend()

    # 子图2: 各特征对前 3 个主成分的总贡献
    loadings = pd.DataFrame({
        'PC1': pca.components_[0],
        'PC2': pca.components_[1],
        'PC3': pca.components_[2]
    }, index=X.columns)
    ax2 = fig.add_subplot(122)
    ax2.barh(range(len(X.columns)), np.abs(loadings).sum(axis=1))
    ax2.set_yticks(range(len(X.columns)))
    ax2.set_yticklabels(X.columns)
    ax2.set_xlabel('对主成分的总贡献度')
    ax2.set_title('各特征对 PCA 前 3 主成分的总贡献')

    plt.suptitle('Wine Quality 数据集 — PCA 3D 降维可视化 + 特征贡献分析', fontsize=14)
    plt.tight_layout()
    plt.show()
    from sklearn.model_selection import cross_val_predict
    model = XGBClassifier(**kwargs)
    model.fit(X.values, y_encoded)
    y_pred = cross_val_predict(model, X.values, y_encoded, cv=5)
    n_errors = sum(y_encoded != y_pred)
    if n_errors > 0:
        print(f'\n5 折交叉验证错误样本: {n_errors}/{len(y_encoded)} = {n_errors/len(y_encoded):.1%}')
    else:
        print('\n5 折交叉验证全部正确！')


"""
XGBoost 完整流水线：训练 → 评估 → 五种可视化。
"""
def run_xgb_pipeline(X_train, X_test, y_train, y_test, X_full, y_encoded, le, **kwargs):
    if 'tree_method' not in kwargs:
        kwargs['tree_method'] = 'hist'
    if 'eval_metric' not in kwargs:
        kwargs['eval_metric'] = 'mlogloss'
    if 'random_state' not in kwargs:
        kwargs['random_state'] = 42
    if 'use_label_encoder' not in kwargs:
        kwargs['use_label_encoder'] = False
    model, y_pred = train_and_evaluate(X_train, X_test, y_train, y_test, le.classes_, **kwargs)
    accuracy = model.score(X_test, y_test)
    plot_confusion_matrix(y_test, y_pred, le.classes_, accuracy)
    plot_feature_importance(model, X_train.columns.tolist())
    plot_pca_decision_boundary(X_full, y_encoded, le.classes_, **kwargs)
    plot_correlation_heatmap(X_full, y_encoded, le.classes_)
    plot_top_features_scatter(X_full, y_encoded, le, **kwargs)
    return model, y_pred
