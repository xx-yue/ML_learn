"""共享单车骑行量预测 — 主入口（Bike Sharing 数据集，11 个特征，3 分类）"""
import os
os.environ['OMP_NUM_THREADS'] = '1'  # 修复 Windows MKL KMeans 内存泄漏（必须在 import sklearn 之前）
import warnings
warnings.filterwarnings('ignore', message='X does not have valid feature names')
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')             # 修复 PyCharm 后端 bug
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ==================== 数据加载 ====================
script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, 'data', 'day.csv')
df = pd.read_csv(file_path)

# 去掉无关列和泄漏列（casual/registered 与 cnt 直接相关）
df = df.drop(columns=['instant', 'dteday', 'casual', 'registered'])

# ==================== 数据预处理 ====================
# 修复中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# 将连续目标 cnt 按三分位数分成 3 类（低/中/高）
df['cnt'] = pd.qcut(df['cnt'], q=3, labels=[0, 1, 2])
df['cnt'] = df['cnt'].astype(int)

X = df.drop('cnt', axis=1)           # 特征矩阵（11 列）
y = df['cnt']                         # 目标向量（0=低, 1=中, 2=高）

# 标签编码
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = ['Low', 'Medium', 'High']  # 低 / 中 / 高骑行量

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# 训练集 / 测试集划分
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f'数据集大小: {X.shape[0]} 条')
print(f'特征数量: {X.shape[1]} 个')
print(f'类别分布: {dict(zip(class_names, pd.Series(y_encoded).value_counts().sort_index().values))}')

# ==================== 1. KNN 训练与可视化 ====================
from components.knn_train import run_knn_pipeline
run_knn_pipeline(X_train, X_test, y_train, y_test, X_scaled, y_encoded, le, n_neighbors=5)

# ==================== 2. 决策树 ====================
# from components.decision_tree import run_dt_pipeline
# run_dt_pipeline(X_train, X_test, y_train, y_test, X_scaled, y_encoded, le,
#                 criterion='gini', max_depth=5)

# ==================== 3. SVM ====================
# from components.svm_train import run_svm_pipeline
# run_svm_pipeline(X_train, X_test, y_train, y_test, X_scaled, y_encoded, le,
#                  kernel='rbf', gamma='scale', C=1.0)

# ==================== 4. 朴素贝叶斯 ====================
# from components.naive_bayes_train import run_nb_pipeline
# run_nb_pipeline(X_train, X_test, y_train, y_test, X_scaled, y_encoded, le)

# ==================== 5. 逻辑回归 ====================
# from components.logistic_regression_train import run_lr_pipeline
# run_lr_pipeline(X_train, X_test, y_train, y_test, X_scaled, y_encoded, le, max_iter=500)

# ==================== 6. 随机森林 ====================
# from components.random_forest_train import run_rf_pipeline
# run_rf_pipeline(X_train, X_test, y_train, y_test, X_scaled, y_encoded, le, n_estimators=100)

# ==================== 7. XGBoost ====================
# from components.xgboost_train import run_xgb_pipeline
# run_xgb_pipeline(X_train, X_test, y_train, y_test, X_scaled, y_encoded, le)

# ==================== 8. LightGBM ====================
# from components.lightgbm_train import run_lgb_pipeline
# run_lgb_pipeline(X_train, X_test, y_train, y_test, X_scaled, y_encoded, le)

# ==================== 9. GBDT ====================
# from components.gbdt_train import run_gbdt_pipeline
# run_gbdt_pipeline(X_train, X_test, y_train, y_test, X_scaled, y_encoded, le, n_estimators=100)

# ==================== 10. AdaBoost ====================
# from components.adaboost_train import run_adaboost_pipeline
# run_adaboost_pipeline(X_train, X_test, y_train, y_test, X_scaled, y_encoded, le, n_estimators=50)

# ==================== 11. K-means ====================
# from components.kmeans_train import run_kmeans_pipeline
# run_kmeans_pipeline(X_scaled.values, y_encoded, n_clusters=3)

# ==================== 12. 密度峰值聚类 ====================
# from components.dpc_train import run_dpc_pipeline
# centers, y_pred_dpc = run_dpc_pipeline(X_scaled.values, y_encoded, n_clusters=3, dc_percent=2.0)
