"""Adult Income 收入预测 — 主入口（Adult 数据集，14 个特征，2 分类）"""
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
column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
                'marital_status', 'occupation', 'relationship', 'race', 'sex',
                'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']
script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, 'data', 'adult.data')
df = pd.read_csv(file_path, names=column_names, skipinitialspace=True)

# ==================== 数据预处理 ====================
# 修复中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# 分类变量编码
categorical_cols = ['workclass', 'education', 'marital_status', 'occupation',
                    'relationship', 'race', 'sex', 'native_country']
for col in categorical_cols:
    le_col = LabelEncoder()
    df[col] = le_col.fit_transform(df[col])

X = df.drop('income', axis=1)       # 特征矩阵（14 列）
y = df['income']                     # 目标向量（<=50K / >50K）

# 标签编码：<=50K→0, >50K→1
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = ['<=50K', '>50K']

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
# run_kmeans_pipeline(X_scaled.values, y_encoded, n_clusters=2)

# ==================== 12. 密度峰值聚类 ====================
# from components.dpc_train import run_dpc_pipeline
# centers, y_pred_dpc = run_dpc_pipeline(X_scaled.values, y_encoded, n_clusters=2, dc_percent=2.0)
