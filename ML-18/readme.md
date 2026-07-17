# ML-18：Online Retail 客户消费分类与聚类

使用 UCI Online Retail 数据集，将交易级数据聚合到客户级别后，对比 **10 种分类算法 + 2 种聚类算法** 的性能与特点。

## 数据集

- 来源：UCI Online Retail 数据集 `data/Online Retail.xlsx`
- 原始数据：541909 条交易记录（英国在线零售商 2010-2011 年）
- 聚合后：约 4372 个客户
- 3 个类别（按总消费三分位数划分）：低消费 / 中消费 / 高消费
- 5 个客户特征：
  1. **recency** — 最近一次购买距今天数
  2. **frequency** — 购买订单数（不同 InvoiceNo 数量）
  3. **num_products** — 购买的不同商品数（不同 StockCode 数量）
  4. **total_quantity** — 购买商品总件数
  5. **avg_order_value** — 平均订单价值（总消费 / 订单数）
- **特征量纲差异大**（recency ~天, total_quantity ~万），必须做 StandardScaler 标准化

## 数据预处理流程

```python
# 1. 去除缺失 CustomerID 的行
df = df.dropna(subset=['CustomerID'])
# 2. 去除退货（负数量）
df = df[df['Quantity'] > 0]
# 3. 计算总价
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
# 4. 聚合到客户级别
customer_df = df.groupby('CustomerID').agg(
    recency=('InvoiceDate', lambda x: (max_date - x.max()).days),
    frequency=('InvoiceNo', 'nunique'),
    num_products=('StockCode', 'nunique'),
    total_quantity=('Quantity', 'sum'),
    total_spending=('TotalPrice', 'sum')
).reset_index()
# 5. 创建平均订单价值
customer_df['avg_order_value'] = customer_df['total_spending'] / customer_df['frequency']
# 6. 按总消费三分位数分为 3 类
customer_df['spending_class'] = pd.qcut(customer_df['total_spending'], q=3, labels=[0, 1, 2])
```

> 注：`total_spending` 从特征中删除（它是分类目标的来源，会导致数据泄露）。

## 项目结构

```
ML-18/
├── main.py                           # 主入口：数据加载 → 聚合 → StandardScaler → 调用各模型
├── data/
│   └── Online Retail.xlsx            # 原始交易数据
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
# 标准化是必须的 —— 5 个特征量纲差异极大
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
```

如果不标准化，total_quantity（~1-196919）等大量纲特征会主导 KNN、SVM 等基于距离的模型。

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
run_kmeans_pipeline(X_scaled.values, y_encoded, n_clusters=3)

# 12. 密度峰值聚类（dc_percent 需调小）
run_dpc_pipeline(X_scaled.values, y_encoded, n_clusters=3, dc_percent=2.0)
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
| **特征重要性** | | **✓** | | | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** |
| **相关性热力图** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **PCA 散点 + 特征贡献** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

> 注：5 维特征使用 5×5 相关性热力图（带数值标注），清晰展示特征间关系。

### 聚类模型

| 可视化 | K-means | 密度峰值聚类 |
|--------|:---:|:---:|
| 聚类报告（ARI / 轮廓系数 / AMI） | ✓ | ✓ |
| PCA 聚类散点图 | ✓ | ✓ |
| 肘部法则图（Elbow） | ✓ | |
| 决策图（γ 排序 + 密度-距离散点图） | | ✓ |

## 依赖

```bash
pip install scikit-learn pandas numpy matplotlib seaborn xgboost lightgbm openpyxl
```

> 注：密度峰值聚类（dpc_train.py）为手写实现，不依赖额外包。`openpyxl` 用于读取 Excel 文件。

## Online Retail vs Wine 对比

| 对比项 | Wine（ML-2） | Online Retail（ML-18） |
|--------|-------------|----------------------|
| 数据来源 | UCI Wine（现成特征矩阵） | UCI Online Retail（需聚合） |
| 样本数 | 178 | ~4372（客户） |
| 特征数 | 13 | 5 |
| 类别数 | 3 | 3 |
| 是否需要标准化 | 必须（量纲差异极大） | **必须**（量纲差异极大） |
| 数据预处理 | 直接读取 | 交易级 → 客户级聚合 |
| 分类目标 | 葡萄酒品种 | 消费等级（三分位数） |
| DPC 参数 | dc_percent=0.1 | dc_percent=2.0 |

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
- [ ] 客户细分业务解读
