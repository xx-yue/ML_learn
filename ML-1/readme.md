# ML-1：鸢尾花分类

使用经典鸢尾花（Iris）数据集，对比 **8 种分类算法** 的性能与特点。

## 数据集

- 来源：UCI 鸢尾花数据集 `data/iris.data`
- 150 条样本，3 个类别（setosa、versicolor、virginica），每类 50 条
- 4 个特征：花萼长度、花萼宽度、花瓣长度、花瓣宽度

## 项目结构

```
ML-1/
├── main.py                         # 主入口：数据加载 → 预处理 → 调用模型
├── data/
│   └── iris.data                   # 数据集
├── components/
│   ├── __init__.py
│   ├── knn_train.py                # K-近邻
│   ├── decision_tree.py            # 决策树
│   ├── svm_train.py                # 支持向量机
│   ├── naive_bayes_train.py        # 朴素贝叶斯
│   ├── logistic_regression_train.py # 逻辑回归
│   ├── random_forest_train.py      # 随机森林
│   ├── xgboost_train.py           # XGBoost
│   └── lightgbm_train.py          # LightGBM
└── readme.md
```

## 模型一览

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

## 运行

在 `main.py` 中取消注释想要运行的模型即可：

```python
# KNN
run_knn_pipeline(X_train, X_test, ..., n_neighbors=3)

# 决策树
run_dt_pipeline(X_train, X_test, ..., criterion='gini', max_depth=3)

# SVM
run_svm_pipeline(X_train, X_test, ..., kernel='rbf', C=1.0)

# 朴素贝叶斯
run_nb_pipeline(X_train, X_test, ...)

# 逻辑回归
run_lr_pipeline(X_train, X_test, ..., max_iter=200)

# 随机森林
run_rf_pipeline(X_train, X_test, ..., n_estimators=100)

# XGBoost
run_xgb_pipeline(X_train, X_test, ..., n_estimators=100)

# LightGBM
run_lgb_pipeline(X_train, X_test, ..., n_estimators=100)
```

然后执行：

```bash
python main.py
```

## 可视化

每个模型均提供以下可视化：

| 可视化 | KNN | 决策树 | SVM | 朴素贝叶斯 | 逻辑回归 | 随机森林 | XGBoost | LightGBM |
|--------|:---:|:------:|:---:|:----------:|:--------:|:--------:|:-------:|:--------:|
| 分类报告 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 混淆矩阵 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| PCA 决策边界 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **树结构图** | | **✓** | | | | | | |
| **特征重要性** | | **✓** | | | | **✓** | **✓** | **✓** |
| Pairplot | ✓ | ✓ | | | | | | |

## 依赖

```bash
pip install sklearn pandas numpy matplotlib seaborn xgboost lightgbm
```

## TODO

- [ ] 网格搜索自动调参
- [ ] 模型性能对比汇总表
- [ ] 交叉验证评估
