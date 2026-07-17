# ML-7：Abalone 鲍鱼年龄分类与聚类

使用 UCI Abalone 数据集，对比 **10 种分类算法 + 2 种聚类算法** 的性能与特点。
Rings（环数）被分箱为 3 个年龄组，Sex 列经 LabelEncoder 编码。

## 数据集

- 来源：UCI Abalone `data/abalone.data`
- 4177 条样本，3 个类别（按环数分箱）
- 8 个特征：Sex（性别）、Length（长度）、Diameter（直径）、Height（高度）、Whole weight（全重）、Shucked weight（去壳重）、Viscera weight（内脏重）、Shell weight（壳重）
- Rings 为目标变量（1-29），分箱为 3 类：0（≤8 环）、1（9-10 环）、2（≥11 环）

## 项目结构

```
ML-7/
├── main.py                           # 主入口：数据加载 → LabelEncoder → 分箱 → StandardScaler → 调用各模型
├── data/
│   └── abalone.data                  # 数据集
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
# Sex 列编码
le_sex = LabelEncoder()
df['Sex'] = le_sex.fit_transform(df['Sex'])

# Rings 分箱为 3 类
df['Rings'] = pd.cut(df['Rings'], bins=[0, 8, 10, 30], labels=[0, 1, 2])
df['Rings'] = df['Rings'].astype(int)
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
