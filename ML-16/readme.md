# ML-16：Adult Income 收入预测与聚类

使用 UCI Adult 数据集，对比 **10 种分类算法 + 2 种聚类算法** 的性能与特点。
二分类问题：预测个人年收入是否超过 50K。

## 数据集

- 来源：UCI Adult 数据集 `data/adult.data`
- 32561 条样本，2 个类别（<=50K / >50K）
- 14 个特征：年龄、工作类别、fnlwgt、教育程度、教育年限、婚姻状况、职业、家庭关系、种族、性别、资本收益、资本损失、每周工时、原国籍
- 8 个分类变量需用 LabelEncoder 编码：workclass, education, marital_status, occupation, relationship, race, sex, native_country

## 项目结构

```
ML-16/
├── main.py                           # 主入口：数据加载 → 分类变量编码 → StandardScaler → 调用各模型
├── data/
│   └── adult.data                    # 数据集
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
# 分类变量编码
categorical_cols = ['workclass', 'education', 'marital_status', 'occupation',
                    'relationship', 'race', 'sex', 'native_country']
for col in categorical_cols:
    le_col = LabelEncoder()
    df[col] = le_col.fit_transform(df[col])

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
