# ML-1：鸢尾花分类

使用经典鸢尾花（Iris）数据集，分别用 **KNN** 和 **决策树** 进行分类任务。

## 数据集

- 来源：UCI 鸢尾花数据集 `data/iris.data`
- 150 条样本，3 个类别（setosa、versicolor、virginica），每类 50 条
- 4 个特征：花萼长度、花萼宽度、花瓣长度、花瓣宽度

## 项目结构

```
ML-1/
├── main.py              # 主入口：数据加载 → 预处理 → 调用模型
├── data/
│   └── iris.data        # 数据集
└── components/
    ├── knn_train.py     # KNN 训练 + 评估 + 可视化
    ├── decision_tree.py # 决策树训练 + 评估 + 可视化
    └── __init__.py
```

## 运行

```bash
cd ML-1
python main.py
```

## 可视化

| 模型 | 可视化 |
|------|--------|
| KNN | 混淆矩阵、PCA 决策边界、特征对散点图 |
| 决策树 | 树结构图、特征重要性、混淆矩阵、PCA 决策边界、特征对散点图 |

## TODO

- [ ] 随机森林
- [ ] 网格搜索调参
- [ ] 模型对比分析
