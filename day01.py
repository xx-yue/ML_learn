from sklearn.datasets import load_iris;
from sklearn.model_selection import train_test_split;
from sklearn.feature_extraction import DictVectorizer;

"""
sklearn数据集使用
:return:
"""
def datasets_demo():
    iris = load_iris()
    # print(type(iris.data))  #ndarrary 高维数组
    print('鸢尾花数据集\n',iris)
    print('查看数据集描述\n',iris['DESCR'])
    print('查看特征值名字\n',iris.feature_names)
    print('查看特征值\n',iris.data,iris.data.shape)
    # print('查看目标值\n',iris.target,iris.target.shape)
    # print('查看目标值(字符串形式)\n',iris.target_names)

    #数据集划分
    x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,test_size=0.2)
    print('训练集特征值和目标值\n',x_train,y_train)
    return None

"""
字典特征抽取
:return:
"""
def dict_demo():
    data=[{'city':'北京','temperature':100},
          {'city':'上海','temperature':60},
          {'city':'深圳','temperature':30}]
    #1.实例化一个转换器类,指定sparse参数为False,不使用稀疏矩阵
    transfer=DictVectorizer(sparse=False)
    #2.调用fit_transform
    data_new=transfer.fit_transform(data)
    print('转换结果\n',data_new)
    print('特征名字\n',transfer.get_feature_names())
    return None


if __name__ == '__main__':
    # datasets_demo()
    dict_demo()