"""鸢尾花分类 — 主入口"""
import os
os.environ['OMP_NUM_THREADS'] = '1'  # 修复 Windows MKL KMeans 内存泄漏
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')             # 修复 PyCharm 后端 bug
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ==================== 数据加载 ====================
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, 'data', 'iris.data')
df = pd.read_csv(file_path, header=None, names=columns)

# ==================== 数据预处理 ====================
# 修复中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

X = df.drop('class', axis=1)       # 特征矩阵
y = df['class']                     # 目标向量

# 标签编码
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 训练集 / 测试集划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ==================== 1. KNN 训练与可视化 ====================
# from components.knn_train import run_knn_pipeline
# run_knn_pipeline(X_train, X_test, y_train, y_test, X, y_encoded, le, n_neighbors=3)

# ==================== 2. 决策树训练与可视化 ====================
# from components.decision_tree import run_dt_pipeline
# run_dt_pipeline(X_train, X_test, y_train, y_test, X, y_encoded, le,
#                 criterion='gini', max_depth=3)

# ==================== 3. SVM 分类器 ====================
# from components.svm_train import run_svm_pipeline
# run_svm_pipeline(X_train, X_test, y_train, y_test, X, y_encoded, le,
#                  kernel='rbf', gamma='scale', C=1.0)

# ==================== 4. 朴素贝叶斯 ====================
# from components.naive_bayes_train import run_nb_pipeline
# run_nb_pipeline(X_train, X_test, y_train, y_test, X, y_encoded, le)

# ==================== 5. 逻辑回归 ====================
# from components.logistic_regression_train import run_lr_pipeline
# run_lr_pipeline(X_train, X_test, y_train, y_test, X, y_encoded, le, max_iter=200)

# ==================== 6. 随机森林 ====================
# from components.random_forest_train import run_rf_pipeline
# run_rf_pipeline(X_train, X_test, y_train, y_test, X, y_encoded, le, n_estimators=100)

# ==================== 7. XGBoost（CPU 模式）====================
# from components.xgboost_train import run_xgb_pipeline
# run_xgb_pipeline(X_train, X_test, y_train, y_test, X, y_encoded, le, tree_method='hist', n_estimators=100)

# ==================== 8. LightGBM（CPU 模式）====================
# from components.lightgbm_train import run_lgb_pipeline
# run_lgb_pipeline(X_train, X_test, y_train, y_test, X, y_encoded, le, boosting_type='gbdt', n_estimators=100)

# ==================== 9. GBDT ====================
# from components.gbdt_train import run_gbdt_pipeline
# run_gbdt_pipeline(X_train, X_test, y_train, y_test, X, y_encoded, le, n_estimators=100)

# ==================== 10. AdaBoost ====================
from components.adaboost_train import run_adaboost_pipeline
# run_adaboost_pipeline(X_train, X_test, y_train, y_test, X, y_encoded, le, n_estimators=50)

# ==================== 11. K-means ====================
from components.kmeans_train import run_kmeans_pipeline
# Iris 去标签跑：X 是全部特征，y_encoded 是原标签（用于算 ARI）
# run_kmeans_pipeline(X.values, y_encoded, n_clusters=3)


# ==================== 12. 密度峰值聚类 ====================
# from components.dpc_train import run_dpc_pipeline
# # DPC 在 Iris 上：dc_percent 调 1.5-3.0 看效果，n_clusters=3
# centers, y_pred_dpc = run_dpc_pipeline(X.values, y_encoded, n_clusters=3, dc_percent=0.5)
