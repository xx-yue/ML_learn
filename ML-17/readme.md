# ML-17：Bike Sharing 共享单车骑行量预测与聚类

使用 UCI Bike Sharing 数据集（day.csv），对比 **10 种分类算法 + 2 种聚类算法** 的性能与特点。
将连续的骑行量 cnt 按三分位数离散化为 3 类（低/中/高），转化为多分类问题。

## 数据集

- 来源：UCI Bike Sharing 数据集 `data/day.csv`
- 731 条样本（按天记录），3 个类别（低/中/高骑行量）
- 11 个特征：季节、年份、月份、是否假期、星期几、是否工作日、天气状况、温度、体感温度、湿度、风速
- 目标 cnt（连续）按 `pd.qcut` 分成 3 类
- 去掉 instant（ID）、dteday（日期）、casual/registered（与 cnt 直接相关，避免数据泄漏）

## 项目结构

```
ML-17/
├── main.py                           # 主入口：数据加载 → 三分位分箱 → StandardScaler → 调用各模型
├── data/
│   └── day.csv                       # 数据集
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
# 去掉无关列和泄漏列
df = df.drop(columns=['instant', 'dteday', 'casual', 'registered'])

# 连续目标 → 3 分类
df['cnt'] = pd.qcut(df['cnt'], q=3, labels=[0, 1, 2])
df['cnt'] = df['cnt'].astype(int)

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
