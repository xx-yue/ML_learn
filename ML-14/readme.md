# ML-14：Banknote Authentication 钞票认证分类与聚类

使用 UCI Banknote Authentication 数据集，对比 **10 种分类算法 + 2 种聚类算法** 的性能与特点。

## 数据集

- 来源：UCI Banknote Authentication 数据集 `data/data_banknote_authentication.txt`
- 1372 条样本，2 个类别（0 = 真钞 Genuine，1 = 伪钞 Forged）
- 4 个特征：variance（方差）、skewness（偏度）、curtosis（峰度）、entropy（熵）
- 目标：class（0/1）→ 二分类
- **特征量纲有差异**，需要 StandardScaler 标准化

## 项目结构

```
ML-14/
├── main.py                           # 主入口：数据加载 → StandardScaler → 调用各模型
├── data/
│   └── data_banknote_authentication.txt  # 数据集（逗号分隔，无表头）
├── components/
│   ├── __init__.py
│   ├── knn_train.py                  # K-近邻
│   ├── decision_tree.py              # 决策树（CART / J48）
│   ├── svm_train.py                  # 支持向量机
│   ├── naive_bayes_train.py          # 朴素贝叶斯
│   ├── logistic_regression_train.py  # 逻辑回归
│   ├── random_forest_train.py        # 随机森林
│   ├── xgboost_train.py              # XGBoost
│   ├── lightgbm_train.py             # LightGBM
│   ├── gbdt_train.py                 # GBDT（梯度提升决策树）
│   ├── adaboost_train.py             # AdaBoost 分类器
│   ├── kmeans_train.py               # K-means 聚类
│   └── dpc_train.py                  # 密度峰值聚类（DPC）
└── readme.md
```

## 预处理要点

```python
# 标准化 — 4 个特征量纲有差异
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
```

## 模型一览

### 分类算法（10 种）

| # | 模型 | 文件 | 类型 |
|---|------|------|------|
| 1 | **KNN** | `knn_train.py` | 基于距离 |
| 2 | **决策树** | `decision_tree.py` | 树模型 |
| 3 | **SVM** | `svm_train.py` | 基于间隔 |
| 4 | **朴素贝叶斯** | `naive_bayes_train.py` | 基于概率 |
| 5 | **逻辑回归** | `logistic_regression_train.py` | 线性分类 |
| 6 | **随机森林** | `random_forest_train.py` | 集成学习（Bagging） |
| 7 | **XGBoost** | `xgboost_train.py` | 集成学习（Boosting） |
| 8 | **LightGBM** | `lightgbm_train.py` | 集成学习（Boosting） |
| 9 | **GBDT** | `gbdt_train.py` | 集成学习（Boosting） |
| 10 | **AdaBoost** | `adaboost_train.py` | 集成学习（Boosting） |

### 聚类算法（2 种）

| # | 模型 | 文件 | 类型 |
|---|------|------|------|
| 11 | **K-means** | `kmeans_train.py` | 划分聚类 |
| 12 | **密度峰值聚类** | `dpc_train.py` | 密度聚类 |

## 运行

在 `main.py` 中取消注释想要运行的模型即可：

```python
# ---------- 分类 ----------
# 1. KNN
run_knn_pipeline(X_train, X_test, ..., n_neighbors=5)

# 2. 决策树
run_dt_pipeline(X_train, X_test, ..., criterion='gini', max_depth=5)

# 3. SVM（二分类需 probability=True 以计算 AUC）
run_svm_pipeline(X_train, X_test, ..., kernel='rbf', C=1.0, probability=True)

# 4. 朴素贝叶斯
run_nb_pipeline(X_train, X_test, ...)

# 5. 逻辑回归
run_lr_pipeline(X_train, X_test, ..., max_iter=500)

# 6. 随机森林
run_rf_pipeline(X_train, X_test, ..., n_estimators=100)

# 7. XGBoost
run_xgb_pipeline(X_train, X_test, ...)

# 8. LightGBM
run_lgb_pipeline(X_train, X_test, ...)

# 9. GBDT
run_gbdt_pipeline(X_train, X_test, ..., n_estimators=100)

# 10. AdaBoost
run_adaboost_pipeline(X_train, X_test, ..., n_estimators=50)

# ---------- 聚类 ----------
# 11. K-means（无监督，去标签运行）
run_kmeans_pipeline(X.values, y_encoded, n_clusters=2)

# 12. 密度峰值聚类
run_dpc_pipeline(X.values, y_encoded, n_clusters=2, dc_percent=2.0)
```

```bash
python main.py
```

## 可视化

### 分类模型

| 可视化 | KNN | 决策树 | SVM | 朴素贝叶斯 | 逻辑回归 | 随机森林 | XGBoost | LightGBM | GBDT | AdaBoost |
|--------|:---:|:------:|:---:|:----------:|:--------:|:--------:|:-------:|:--------:|:----:|:--------:|
| 分类报告 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| AUC | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 混淆矩阵 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| PCA 决策边界 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **特征重要性** | | **✓** | | | **✓**(coef_) | **✓** | **✓** | **✓** | **✓** | **✓** |
| **相关性热力图** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **PCA 散点 + 特征贡献** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

> 注：二分类使用 coolwarm 配色，决策边界 colorbar ticks=[0,1]。

### 聚类模型

| 可视化 | K-means | 密度峰值聚类 |
|--------|:---:|:---:|
| 聚类报告（ARI / 轮廓系数 / AMI） | ✓ | ✓ |
| PCA 聚类散点图 | ✓ | ✓ |
| 肘部法则图（Elbow） | ✓ | |
| 决策图（γ 排序 + 密度-距离散点图） | | ✓ |

## 依赖

```bash
pip install scikit-learn pandas numpy matplotlib seaborn xgboost lightgbm
```

> 注：密度峰值聚类（dpc_train.py）为手写实现，不依赖额外包。
