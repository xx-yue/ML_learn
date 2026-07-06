from  sklearn.datasets import load_iris;
from sklearn.model_selection import train_test_split;
from  sklearn.preprocessing import StandardScaler;
from sklearn.neighbors import KNeighborsClassifier;
from sklearn.metrics import classification_report;


"""
Knn 近邻算法 对鸢尾花进行分类
:return:
"""
def knn_iris():
    #1. 获取数据
    iris = load_iris()
    #2. 划分数据集
    x_train,x_test,y_train,y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=6)
    #3. 特征工程:标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    #4. 机器学习(模型训练)
    estimator = KNeighborsClassifier(n_neighbors=3)
    estimator.fit(x_train, y_train)
    #5. 模型评估
    # 方法1 直接对比真实值和预测值
    y_pred = estimator.predict(x_test)
    print("y_predict:\n", y_pred)
    print('直接对比真实值：\n', classification_report(y_test, y_pred))

    #方法2 计算准确率
    score = estimator.score(x_test, y_test)
    print('准确率为 \n',score)

    return  None

if __name__ == '__main__':
    # KNN 对鸢尾花进行分类
    knn_iris()