from sklearn.datasets import load_iris;
from sklearn.model_selection import train_test_split;
from sklearn.feature_extraction import DictVectorizer;
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer;
from sklearn.preprocessing import MinMaxScaler,StandardScaler;
import jieba;
import pandas as pd;

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

"""
文本特征抽取
:return:
"""
def count_demo():
    data=['life is short,i like python','life is is too long,i love python']
    #1.实例化一个转换器类
    transfer=CountVectorizer()
    #2.调用fit_transform
    data_new=transfer.fit_transform(data)
    print('文本特征抽取\n',data_new.toarray())
    print('特征名字\n',transfer.get_feature_names())
    return None

"""
中文文本特征抽取
:return:
"""
def cut_word(text):
    it=jieba.cut(text)
    return ' '.join(it)
def count_chinese_demo():
    data=['一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。',
          '我们看到的从很远星系来的光是在几百万年前发出的，这样当我们看到宇宙时，我们是在看它的过去。',
          '如果只用一种方式了解某样事物，你就不会真正了解它。了解实物真正含义的秘密取决于如何将其与我们所了解的事物相联系。'
          ]
    data_new = []
    for d in data:
        data_new.append(cut_word(d))
    #1.实例化一个转换器类
    transfer=CountVectorizer()
    #2.调用fit_transform
    data_new2=transfer.fit_transform(data_new)
    print('文本特征抽取\n',data_new2.toarray())
    print('特征名字\n',transfer.get_feature_names())
    return None

"""
使用TF-idf进行文本特征抽取
:return:
"""
def tf_idf_demo():
    data=['一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。',
          '我们看到的从很远星系来的光是在几百万年前发出的，这样当我们看到宇宙时，我们是在看它的过去。',
          '如果只用一种方式了解某样事物，你就不会真正了解它。了解实物真正含义的秘密取决于如何将其与我们所了解的事物相联系。'
          ]
    data_new = []
    for d in data:
        data_new.append(cut_word(d))
    #1.实例化一个转换器类
    transfer=TfidfVectorizer()
    #2.调用fit_transform
    data_new2=transfer.fit_transform(data_new)
    print('文本特征抽取\n',data_new2.toarray())
    print('特征名字\n',transfer.get_feature_names())
    return None

"""
归一化
:return:
"""
def min_max_demo():
    data = pd.read_csv('./data/dating.txt')
    data = data.iloc[:,:3]
    # 实例化一个转换器类
    transfer = MinMaxScaler(feature_range=[2,3])
    # 调用fit_transform
    data_new = transfer.fit_transform(data)

    print('最小最大值归一化\n',data_new)
    return None


"""
标准化
:return:
"""
def stand_demo():
    data = pd.read_csv('./data/dating.txt')
    data = data.iloc[:,:3]
    # 实例化一个转换器类
    transfer = StandardScaler()
    # 调用fit_transform
    data_new = transfer.fit_transform(data)

    print('最小最大值归一化\n',data_new)
    return None


if __name__ == '__main__':
    # datasets_demo()
    # dict_demo()
    # count_demo()
    # count_chinese_demo()
    # tf_idf_demo()
    # min_max_demo()
    stand_demo()