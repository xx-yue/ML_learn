"""Online Retail 客户消费分类与聚类 — 主入口（5 维客户特征，3 类消费等级）"""
import os
os.environ['OMP_NUM_THREADS'] = '1'  # 修复 Windows MKL KMeans 内存泄漏（必须在 import sklearn 之前）
import warnings
warnings.filterwarnings('ignore', message='X does not have valid feature names')
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')             # 修复 PyCharm 后端 bug
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ==================== 数据加载 ====================
script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, 'data', 'Online Retail.xlsx')
df = pd.read_excel(file_path)

# 去除缺失 CustomerID 的行
df = df.dropna(subset=['CustomerID'])
# 去除退货（负数量）
df = df[df['Quantity'] > 0]
# 计算总价
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
# 转换日期
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# ==================== 聚合到客户级别 ====================
max_date = df['InvoiceDate'].max()
customer_df = df.groupby('CustomerID').agg(
    recency=('InvoiceDate', lambda x: (max_date - x.max()).days),
    frequency=('InvoiceNo', 'nunique'),
    num_products=('StockCode', 'nunique'),
    total_quantity=('Quantity', 'sum'),
    total_spending=('TotalPrice', 'sum')
).reset_index()

# 平均订单价值
customer_df['avg_order_value'] = customer_df['total_spending'] / customer_df['frequency']

# 去掉 CustomerID
customer_df = customer_df.drop(columns=['CustomerID'])

# 按总消费的三分位数分为 3 类（分类目标）
customer_df['spending_class'] = pd.qcut(customer_df['total_spending'], q=3, labels=[0, 1, 2])
customer_df['spending_class'] = customer_df['spending_class'].astype(int)

# ==================== 数据预处理 ====================
# 修复中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# 特征：recency, frequency, num_products, total_quantity, avg_order_value（5 个特征）
# 目标：spending_class（0=低消费, 1=中消费, 2=高消费）
# 从特征中删除 total_spending（它是目标来源，会导致数据泄露）
X = customer_df.drop(columns=['spending_class', 'total_spending'])
y = customer_df['spending_class']

# y 已通过 qcut 编码为 0/1/2，无需 LabelEncoder 转换
y_encoded = y.values
class_names = ['低消费', '中消费', '高消费']  # 3 类消费等级
le = LabelEncoder()
le.classes_ = np.array(class_names)

# 标准化 — 客户特征量纲差异大（recency ~天, total_quantity ~万），必须标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# 训练集 / 测试集划分
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f'数据集大小: {X.shape[0]} 条客户')
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
# run_gbdt_pipeline(X_train, X_test, y_train, y_test, X, y_encoded, le, n_estimators=100)

# ==================== 10. AdaBoost ====================
# from components.adaboost_train import run_adaboost_pipeline
# run_adaboost_pipeline(X_train, X_test, y_train, y_test, X, y_encoded, le, n_estimators=50)

# ==================== 11. K-means ====================
# from components.kmeans_train import run_kmeans_pipeline
# run_kmeans_pipeline(X_scaled.values, y_encoded, n_clusters=3)

# ==================== 12. 密度峰值聚类 ====================
# from components.dpc_train import run_dpc_pipeline
# centers, y_pred_dpc = run_dpc_pipeline(X_scaled.values, y_encoded, n_clusters=3, dc_percent=2.0)
