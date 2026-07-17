# ML-10：Concrete 混凝土抗压强度分类与聚类

使用 UCI Concrete Compressive Strength 数据集，对比 **10 种分类算法 + 2 种聚类算法** 的性能与特点。
关键特点：8 维特征 + 回归目标离散化为 3 分类问题。

## 数据集

- 来源：UCI Concrete 数据集 `data/Concrete_Data.xls`
- 1030 条样本，3 个类别（抗压强度按三分位数离散化：Low / Medium / High）
- 8 个特征：Cement（水泥）, Slag（矿渣）, Ash（粉煤灰）, Water（水）,
  Superplasticizer（减水剂）, CoarseAgg（粗骨料）, FineAgg（细骨料）, Age（龄期）
- **特征量纲差异大**，必须做 StandardScaler 标准化

## 项目结构

```
ML-10/
├── main.py                           # 主入口：数据加载 → 离散化 → StandardScaler → 调用各模型
├── data/
│   └── Concrete_Data.xls             # 数据集（Excel 格式）
├── components/
│   ├── __init__.py
│   ├── knn_train.py                  # K-近邻
│   ├── decision_tree.py              # 决策树
│   ├── svm_train.py                  # 支持向量机
│   ├── naive_bayes_train.py          # 朴素贝叶斯
│   ├── logistic_regression_train.py  # 逻辑回归
│   ├── random_forest_train.py        # 随机森林
│   ├── xgboost_train.py              # XGBoost
│   ├── lightgbm_train.py             # LightGBM
│   ├── gbdt_train.py                 # GBDT
│   ├── adaboost_train.py             # AdaBoost
│   ├── kmeans_train.py               # K-means 聚类
│   └── dpc_train.py                  # 密度峰值聚类（DPC）
└── readme.md
```

## 预处理要点

```python
# 1. 读取 Excel 并重命名列
df = pd.read_excel('data/Concrete_Data.xls')
df.columns = ['Cement', 'Slag', 'Ash', 'Water', 'Superplasticizer',
              'CoarseAgg', 'FineAgg', 'Age', 'Strength']

# 2. 将连续目标按三分位数离散为 3 类
df['Strength'] = pd.qcut(df['Strength'], q=3, labels=[0, 1, 2])
df['Strength'] = df['Strength'].astype(int)

# 3. 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
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

```bash
python main.py
```

## 多分类适配

- **cmap**: `Set1`（多色，适合 3 分类可视化）
- **colorbar ticks**: `[0, 1, 2]`
- **相关性热力图**: 8 个特征，`annot=True`，`figsize=(12,10)`

## 依赖

```bash
pip install scikit-learn pandas numpy matplotlib seaborn xgboost lightgbm openpyxl
```

## TODO

- [ ] 网格搜索自动调参
- [ ] 模型性能对比汇总表
- [ ] 交叉验证评估
- [ ] 回归 vs 分类对比（原始连续目标）
