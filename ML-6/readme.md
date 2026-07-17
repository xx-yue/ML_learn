# ML-6：Car Evaluation 汽车评估分类与聚类

使用 UCI Car Evaluation 数据集，对比 **10 种分类算法 + 2 种聚类算法** 的性能与特点。
所有 6 个特征均为分类型变量，经 LabelEncoder 编码后输入模型。

## 数据集

- 来源：UCI Car Evaluation `data/car.data`
- 1728 条样本，4 个类别（unacc / acc / good / vgood），分布不均衡
- 6 个特征：buying（购买价格）、maint（维护费用）、doors（车门数）、persons（载人数）、lug_boot（行李厢大小）、safety（安全性）
- **全部 6 个特征均为分类型字符串**，需用 LabelEncoder 编码

## 项目结构

```
ML-6/
├── main.py                           # 主入口：数据加载 → LabelEncoder → StandardScaler → 调用各模型
├── data/
│   └── car.data                      # 数据集
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
# 所有列（含目标）均为分类字符串，统一 LabelEncoder 编码
for col in df.columns:
    le_col = LabelEncoder()
    df[col] = le_col.fit_transform(df[col])
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

## TODO

- [ ] 网格搜索自动调参
- [ ] 模型性能对比汇总表
- [ ] 交叉验证评估
