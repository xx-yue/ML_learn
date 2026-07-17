# ML-11：Auto MPG 汽车燃油效率分类与聚类

使用 UCI Auto MPG 数据集，对比 **10 种分类算法 + 2 种聚类算法** 的性能与特点。
关键特点：7 维特征 + 回归目标离散化为 3 分类问题。

## 数据集

- 来源：UCI Auto MPG 数据集 `data/auto-mpg.data`
- ~392 条样本（去除 horsepower 缺失值后），3 个类别（MPG 按三分位数离散化：Low / Medium / High）
- 7 个特征：cylinders（气缸数）, displacement（排量）, horsepower（马力）,
  weight（车重）, acceleration（加速）, model_year（车型年份）, origin（产地）
- **horsepower 列含 '?' 缺失值**，需 `pd.to_numeric(errors='coerce')` 后 `dropna()`
- **特征量纲差异大**，必须做 StandardScaler 标准化

## 项目结构

```
ML-11/
├── main.py                           # 主入口：数据加载 → 缺失值处理 → 离散化 → StandardScaler → 调用各模型
├── data/
│   └── auto-mpg.data                 # 数据集（空格分隔文本格式）
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
# 1. 读取数据（空格分隔，无表头）
column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
                'acceleration', 'model_year', 'origin', 'car_name']
df = pd.read_csv('data/auto-mpg.data', delim_whitespace=True, names=column_names)

# 2. 处理 horsepower 缺失值（'?' → NaN → dropna）
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
df = df.dropna()
df = df.drop('car_name', axis=1)

# 3. 将连续目标 mpg 按三分位数离散为 3 类
df['mpg'] = pd.qcut(df['mpg'], q=3, labels=[0, 1, 2])
df['mpg'] = df['mpg'].astype(int)

# 4. 标准化
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
- **相关性热力图**: 7 个特征，`annot=True`，`figsize=(12,10)`

## 依赖

```bash
pip install scikit-learn pandas numpy matplotlib seaborn xgboost lightgbm
```

## TODO

- [ ] 网格搜索自动调参
- [ ] 模型性能对比汇总表
- [ ] 交叉验证评估
- [ ] 回归 vs 分类对比（原始连续目标）
