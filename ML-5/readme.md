# ML-5：葡萄酒质量分类与聚类

使用 UCI Wine Quality（红葡萄酒）数据集，对比 **10 种分类算法 + 2 种聚类算法** 的性能与特点。
与 ML-2 的关键区别在于：11 维理化指标 + **6 个质量等级** + 1599 条样本 + 严重的类别不平衡。

## 数据集

- 来源：UCI Wine Quality 数据集 `data/winequality-red.csv`（仅红葡萄酒）
- 1599 条样本，6 个质量等级（quality 3-8），分布严重不均衡（5 和 6 占绝大多数）
- 11 个特征：固定酸度、挥发酸、柠檬酸、残糖、氯化物、游离二氧化硫、总二氧化硫、密度、pH、硫酸盐、酒精度
- CSV 格式，**分号分隔**，带表头
- **特征量纲差异大**（如二氧化硫 ~100、密度 ~1），必须做 StandardScaler 标准化

## 项目结构

```
ML-5/
├── main.py                           # 主入口：数据加载 → StandardScaler → 调用各模型
├── data/
│   ├── winequality-red.csv            # 数据集（分号分隔）
│   └── winequality.names              # 数据描述文件
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
# 分号分隔读取
df = pd.read_csv('data/winequality-red.csv', sep=';')

# 标准化是必须的 —— 11 个特征量纲差异大
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
```

如果不标准化，总二氧化硫（~100）等大量纲特征会主导 KNN、SVM 等基于距离的模型。

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
# 1. KNN（标准化后 k 默认 5）
run_knn_pipeline(X_train, X_test, ..., n_neighbors=5)

# 2. 决策树
run_dt_pipeline(X_train, X_test, ..., criterion='gini', max_depth=5)

# 3. SVM
run_svm_pipeline(X_train, X_test, ..., kernel='rbf', C=1.0)

# 4. 朴素贝叶斯
run_nb_pipeline(X_train, X_test, ...)

# 5. 逻辑回归（max_iter 需调大）
run_lr_pipeline(X_train, X_test, ..., max_iter=500)

# 6. 随机森林
run_rf_pipeline(X_train, X_test, ..., n_estimators=100)

# 7. XGBoost（CPU 模式）
run_xgb_pipeline(X_train, X_test, ...)

# 8. LightGBM（CPU 模式）
run_lgb_pipeline(X_train, X_test, ...)

# 9. GBDT
run_gbdt_pipeline(X_train, X_test, ..., n_estimators=100)

# 10. AdaBoost
run_adaboost_pipeline(X_train, X_test, ..., n_estimators=50)

# ---------- 聚类 ----------
# 11. K-means（无监督，去标签运行）
run_kmeans_pipeline(X.values, y_encoded, n_clusters=6)

# 12. 密度峰值聚类（dc_percent 需调小）
run_dpc_pipeline(X.values, y_encoded, n_clusters=6, dc_percent=0.1)
```

```bash
python main.py
```

## 可视化

### 分类模型

| 可视化 | KNN | 决策树 | SVM | 朴素贝叶斯 | 逻辑回归 | 随机森林 | XGBoost | LightGBM | GBDT | AdaBoost |
|--------|:---:|:------:|:---:|:----------:|:--------:|:--------:|:-------:|:--------:|:----:|:--------:|
| 分类报告 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 混淆矩阵 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| PCA 决策边界 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **特征重要性** | | **✓** | | | | **✓** | **✓** | **✓** | **✓** | **✓** |
| **相关性热力图** | ✓ | ✓ | | | | | | | | |
| **PCA 散点 + 特征贡献** | ✓ | ✓ | | | | | | | | |

> 注：11 维特征无法使用 Pairplot，改用 **11×11 相关性热力图** 展示特征间关系。

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

## Wine Quality vs Wine vs Iris 对比

| 对比项 | Iris（ML-1） | Wine（ML-2） | Wine Quality（ML-5） |
|--------|-------------|-------------|-------------|
| 样本数 | 150 | 178 | **1599** |
| 特征数 | 4 | 13 | **11** |
| 类别数 | 3 | 3 | **6** |
| 是否需要标准化 | 不需要（量纲相近） | 必须（量纲差异极大） | **必须**（量纲差异大） |
| 类别不平衡 | 无（各 50） | 轻微（59/71/48） | **严重**（5 和 6 占绝大多数） |
| 分隔符 | 逗号 | 逗号 | **分号** |
| 决策边界配色 | Set1（3色） | Set1（3色） | **Set1（6色）** |

## 算法覆盖对照（导师清单）

| 清单类别 | 要求 | 已实现 | 状态 |
|----------|------|--------|:---:|
| 分类 | KNN / SVM / CART | KNN、SVM、决策树 | ✅ |
| 分类 | 朴素贝叶斯、逻辑回归 | 朴素贝叶斯、逻辑回归 | ✅ 加分项 |
| 集成学习 | RandomForest / XGB / GBDT / AdaBoost / LGBM | 全部 5 种 | ✅ |
| 聚类 | K-means / 密度峰值聚类 | 全部 2 种 | ✅ |

## TODO

- [ ] 网格搜索自动调参
- [ ] 模型性能对比汇总表
- [ ] 交叉验证评估
- [ ] 处理类别不平衡（SMOTE / class_weight）
- [ ] 回归任务（将质量分数视为连续值）
