"""LightGBM 分类器 — Online Retail 数据集（5 维客户特征）训练、评估与可视化（CPU 模式）"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA



"""训练 LightGBM 分类器并打印分类报告。

基于梯度提升框架，采用 Leaf-wise 分裂策略，
比 XGBoost 训练更快、内存更省，但在小数据上可能过拟合。

Parameters
----------
**kwargs : dict
    传递给 LGBMClassifier 的参数（如 boosting_type、n_estimators 等）

Returns
-------
model : LGBMClassifier
    训练好的模型
y_pred : ndarray
    测试集预测结果"""
def train_and_evaluate(X_train, X_test, y_train, y_test, class_names, **kwargs):
    model = LGBMClassifier(**kwargs)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    class_names = [str(c) for c in class_names]
    print(f'\n{"="*60}')
    print(f'  LightGBM 分类报告 (n_estimators={kwargs.get("n_estimators",100)})')
    print(f'{"="*60}')
    print(classification_report(y_test, y_pred, target_names=class_names))
    return model, y_pred



"""绘制混淆矩阵热力图，展示真实类别 vs 预测类别的分布。"""
def plot_confusion_matrix(y_test, y_pred, class_names, accuracy):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='YlGnBu', values_format='d')
    plt.title(f'混淆矩阵 — LightGBM\n准确率: {accuracy:.2%}')
    plt.tight_layout(); plt.show()



"""绘制特征重要性柱状图。LightGBM 基于梯度提升过程中的特征分裂次数和增益总和计算贡献度。"""
def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10,6))
    bars = plt.bar(range(len(importances)), importances[indices],
                   color=['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00'][:len(importances)])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices],
               rotation=45, ha='right')
    plt.ylabel('重要性')
    plt.title('特征重要性 (LightGBM)')
    for bar, val in zip(bars, importances[indices]):
        plt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                 f'{val:.3f}', ha='center', fontsize=9)
    plt.tight_layout(); plt.show()


"""PCA 降维到 2D，在降维后空间训练 LightGBM 并绘制决策边界。"""
def plot_pca_decision_boundary(X, y, class_names, **kwargs):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.2, random_state=42, stratify=y)
    model = LGBMClassifier(**kwargs)
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
    plt.title(f'LightGBM 决策边界\n准确率: {model.score(X_test,y_test):.2%}')
    plt.colorbar(scatter, ticks=[0,1,2], label='类别')
    plt.tight_layout(); plt.show()



"""绘制 5×5 特征相关性热力图，并打印各类别特征均值，辅助分析哪些成分对分类影响最大。"""
def plot_correlation_heatmap(X, y_encoded, class_names):
    df_corr = X.copy()
    df_corr['类别'] = y_encoded
    print(f'\n{"="*60}\n  各类别特征均值对比\n{"="*60}')
    for i, name in enumerate(class_names):
        print(f'\n--- {name} ---')
        print(df_corr[df_corr['类别']==i].drop('类别',axis=1).mean().to_string())
    plt.figure(figsize=(10,8))
    corr = X.corr()
    mask = np.triu(np.ones_like(corr,dtype=bool),k=1)
    sns.heatmap(corr,mask=mask,annot=True,fmt='.2f',cmap='RdBu_r',
                center=0,vmin=-1,vmax=1,square=True,linewidths=0.5,
                cbar_kws={'shrink':0.8,'label':'相关系数'})
    plt.title('Online Retail 数据集 5 个特征相关性矩阵',fontsize=14)
    plt.xticks(rotation=45,ha='right'); plt.yticks(rotation=0)
    plt.tight_layout(); plt.show()


"""PCA 降维双图：左图展示样本在 2D 空间的分布，右图展示各特征对主成分的贡献度，最后用 5 折交叉验证评估泛化能力。"""
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
    plt.suptitle('Online Retail 数据集 — PCA 降维 + 特征贡献',fontsize=14)
    plt.tight_layout(); plt.show()
    from sklearn.model_selection import cross_val_predict
    model = LGBMClassifier(**kwargs)
    model.fit(X.values, y_encoded)  # 转 numpy 避免特征名警告
    y_pred = cross_val_predict(model, X.values, y_encoded, cv=5)
    n_errors = sum(y_encoded != y_pred)
    if n_errors > 0:
        print(f'\n5 折交叉验证错误样本: {n_errors}/{len(y_encoded)} = {n_errors/len(y_encoded):.1%}')
    else:
        print('\n5 折交叉验证全部正确！')




"""LightGBM 完整流水线：训练 → 评估 → 混淆矩阵 → 特征重要性 → PCA 决策边界 → 相关性热力图 → PCA 散点 + 交叉验证。

默认强制 CPU 模式（boosting_type='gbdt'），关闭训练日志。
供 main.py 调用的统一入口。"""
def run_lgb_pipeline(X_train, X_test, y_train, y_test, X_full, y_encoded, le, **kwargs):
    if 'boosting_type' not in kwargs:
        kwargs['boosting_type'] = 'gbdt'
    if 'random_state' not in kwargs:
        kwargs['random_state'] = 42
    if 'verbose' not in kwargs:
        kwargs['verbose'] = -1
    model, y_pred = train_and_evaluate(X_train, X_test, y_train, y_test, le.classes_, **kwargs)
    accuracy = model.score(X_test, y_test)
    plot_confusion_matrix(y_test, y_pred, le.classes_, accuracy)
    plot_feature_importance(model, X_train.columns.tolist())
    plot_pca_decision_boundary(X_full, y_encoded, le.classes_, **kwargs)
    plot_correlation_heatmap(X_full, y_encoded, le.classes_)
    plot_top_features_scatter(X_full, y_encoded, le, **kwargs)
    return model, y_pred
