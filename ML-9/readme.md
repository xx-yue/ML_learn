# ML-9：Bank Marketing 银行营销预测与聚类

使用 UCI Bank Marketing 数据集，对比 **10 种分类算法 + 2 种聚类算法** 的性能与特点。
关键特点：20 维混合类型特征 + 大规模样本（41188 条）+ 二分类不平衡问题。

## 数据集

- 来源：UCI Bank Marketing 数据集 `data/bank-additional/bank-additional-full.csv`
- 41188 条样本，2 个类别（客户是否订阅定期存款：yes/no），分布不均衡
- 20 个特征：age, job, marital, education, default, housing, loan, contact, month, day_of_week,
  duration, campaign, pdays, previous, poutcome, emp.var.rate, cons.price.idx, cons.conf.idx,
  euribor3m, nr.employed
- **10 个分类特征** 需 LabelEncoder 编码：job, marital, education, default, housing, loan,
  contact, month, day_of_week, poutcome
- **特征量纲差异大**，必须做 StandardScaler 标准化

## 项目结构

```
ML-9/
├── main.py                           # 主入口：数据加载 → 编码 → StandardScaler → 调用各模型
├── data/
│   └── bank-additional/
│       └── bank-additional-full.csv  # 数据集（分号分隔）
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
# 1. 读取分号分隔的 CSV
df = pd.read_csv('data/bank-additional/bank-additional-full.csv', sep=';')

# 2. 编码 10 个分类列
categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan',
                    'contact', 'month', 'day_of_week', 'poutcome']
for col in categorical_cols:
    le_col = LabelEncoder()
    df[col] = le_col.fit_transform(df[col])

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

## 二分类适配

- **cmap**: `coolwarm`（红蓝双色，适合二分类可视化）
- **colorbar ticks**: `[0, 1]`
- **AUC**: 对支持 `predict_proba` 的模型（决策树、逻辑回归、随机森林、XGBoost、LightGBM、GBDT、AdaBoost）计算 ROC AUC
- **相关性热力图**: 20 个特征，`annot=False`，`figsize=(16,14)`

## 依赖

```bash
pip install scikit-learn pandas numpy matplotlib seaborn xgboost lightgbm openpyxl
```

## TODO

- [ ] 网格搜索自动调参
- [ ] 模型性能对比汇总表
- [ ] 交叉验证评估
- [ ] 处理类别不平衡（SMOTE / class_weight）
