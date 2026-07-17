# ML-15：Mammographic Masses 乳腺肿块恶性预测与聚类

使用 UCI Mammographic Masses 数据集，对比 **10 种分类算法 + 2 种聚类算法** 的性能与特点。
二分类问题：预测乳腺肿块是良性（Benign）还是恶性（Malignant）。

## 数据集

- 来源：UCI Mammographic Masses 数据集 `data/mammographic_masses.data`
- 原始 961 条样本，去除缺失值后约 830 条
- 2 个类别：良性（Benign）/ 恶性（Malignant）
- 5 个特征：BI_RADS 评估等级、年龄、形状、边缘、密度
- 缺失值用 `?` 标记，读取时设 `na_values='?'`，然后 `dropna()`

## 项目结构

```
ML-15/
├── main.py                           # 主入口：数据加载 → StandardScaler → 调用各模型
├── data/
│   └── mammographic_masses.data      # 数据集
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
# 缺失值处理
df = pd.read_csv('data/mammographic_masses.data', names=column_names, na_values='?')
df = df.dropna()

# 标准化
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

## 依赖

```bash
pip install scikit-learn pandas numpy matplotlib seaborn xgboost lightgbm
```
