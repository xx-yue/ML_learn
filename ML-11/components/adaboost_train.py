"""AdaBoost 分类器 — Auto MPG 数据集（7 维特征）训练、评估与可视化"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA


"""
训练 AdaBoost 并打印分类报告。
"""
def train_and_evaluate(X_train, X_test, y_train, y_test, class_names, **kwargs):
    model = AdaBoostClassifier(**kwargs)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    class_names = [str(c) for c in class_names]
    print(f'\n{"="*60}')
    print(f'  AdaBoost 分类报告 (n_estimators={kwargs.get("n_estimators",50)})')
    print(f'{"="*60}')
    print(classification_report(y_test, y_pred, target_names=class_names))
    return model, y_pred


"""
绘制混淆矩阵热力图。
"""
def plot_confusion_matrix(y_test, y_pred, class_names, accuracy):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Wistia', values_format='d')
    plt.title(f'混淆矩阵 — AdaBoost\n准确率: {accuracy:.2%}')
    plt.tight_layout(); plt.show()


"""
绘制 AdaBoost 特征重要性条形图。
"""
def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10,6))
    bars = plt.bar(range(len(importances)), importances[indices],
                   color=['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00',
                          '#ffff33','#a65628'][:len(importances)])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices],
               rotation=45, ha='right')
    plt.ylabel('重要性')
    plt.title('特征重要性 (AdaBoost)')
    for bar, val in zip(bars, importances[indices]):
        plt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                 f'{val:.3f}', ha='center', fontsize=9)
    plt.tight_layout(); plt.show()


"""
PCA 降维到 2D，绘制 AdaBoost 决策边界。
"""
def plot_pca_decision_boundary(X, y, class_names, **kwargs):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.2, random_state=42, stratify=y)
    model = AdaBoostClassifier(**kwargs)
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
    plt.title(f'AdaBoost 决策边界\n准确率: {model.score(X_test,y_test):.2%}')
    plt.colorbar(scatter, ticks=[0,1,2], label='类别')
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
    plt.figure(figsize=(12,10))
    corr = X.corr()
    mask = np.triu(np.ones_like(corr,dtype=bool),k=1)
    sns.heatmap(corr,mask=mask,annot=True,fmt='.2f',cmap='RdBu_r',
                center=0,vmin=-1,vmax=1,square=True,linewidths=0.5,
                cbar_kws={'shrink':0.8,'label':'相关系数'})
    plt.title('Auto MPG 数据集 7 个特征相关性矩阵',fontsize=14)
    plt.xticks(rotation=45,ha='right'); plt.yticks(rotation=0)
    plt.tight_layout(); plt.show()


"""
PCA 散点图 + 特征贡献分析。
"""
def plot_top_features_scatter(X, y_encoded, le, **kwargs):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    fig, axes = plt.subplots(1,2,figsize=(16,6))
    for i, name in enumerate(le.classes_):
        mask = y_encoded == i
        axes[0].scatter(X_pca[mask,0],X_pca[mask,1],label=name,alpha=0.7,edgecolors='k',s=50)
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    axes[0].set_title(f'PCA 散点图 (累计方差: {pca.explained_variance_ratio_.sum():.1%})')
    axes[0].legend()
    loadings = pd.DataFrame({'PC1':pca.components_[0],'PC2':pca.components_[1]}, index=X.columns)
    axes[1].barh(range(len(X.columns)), np.abs(loadings).sum(axis=1))
    axes[1].set_yticks(range(len(X.columns)))
    axes[1].set_yticklabels(X.columns)
    axes[1].set_xlabel('总贡献度')
    axes[1].set_title('特征对前2主成分的贡献')
    plt.suptitle('Auto MPG 数据集 — PCA 降维 + 特征贡献',fontsize=14)
    plt.tight_layout(); plt.show()
    from sklearn.model_selection import cross_val_predict
    model = AdaBoostClassifier(**kwargs)
    model.fit(X.values, y_encoded)
    y_pred = cross_val_predict(model, X.values, y_encoded, cv=5)
    n_errors = sum(y_encoded != y_pred)
    if n_errors > 0:
        print(f'\n5 折交叉验证错误样本: {n_errors}/{len(y_encoded)} = {n_errors/len(y_encoded):.1%}')
    else:
        print('\n5 折交叉验证全部正确！')


"""
AdaBoost 完整流水线：训练 → 评估 → 五种可视化。
"""
def run_adaboost_pipeline(X_train, X_test, y_train, y_test, X_full, y_encoded, le, **kwargs):
    if 'n_estimators' not in kwargs:
        kwargs['n_estimators'] = 50
    if 'estimator' not in kwargs:
        kwargs['estimator'] = DecisionTreeClassifier(max_depth=1)
    if 'random_state' not in kwargs:
        kwargs['random_state'] = 42
    model, y_pred = train_and_evaluate(X_train, X_test, y_train, y_test, le.classes_, **kwargs)
    accuracy = model.score(X_test, y_test)
    plot_confusion_matrix(y_test, y_pred, le.classes_, accuracy)
    plot_feature_importance(model, X_train.columns.tolist())
    plot_pca_decision_boundary(X_full, y_encoded, le.classes_, **kwargs)
    plot_correlation_heatmap(X_full, y_encoded, le.classes_)
    plot_top_features_scatter(X_full, y_encoded, le, **kwargs)
    return model, y_pred
