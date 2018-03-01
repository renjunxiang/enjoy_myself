import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, scorer


class stockmodel():
    def data_transform(self, data, type='delta', length=30):
        '''
        
        :param data: list
        :param type: 'delta' or 'raw
        :param length: 截取的变量长度
        :return: 
        '''
        # 转为差分数据
        if type == 'delta':
            data = data
        else:
            data = np.diff(data)

        # 生成标签
        y = data[length:]
        y = np.where(np.array(y) > 0, 'up', 'down')

        # 生成数据
        x = []
        for i in range(len(data) - length):
            x.append(data[i:i + length])
        self.x = data[-length:]
        return x, y

    def model_train(self, x, y, method='Logistic', **param):
        '''
        
        :param x: 训练数集
        :param y: 训练标签集
        :param method:模型名称 
        :param param: 模型可变参数
        :return: 
        '''
        if method == 'Logistic':
            model = LogisticRegression(**param)
        elif method == 'KNN':
            model = KNeighborsClassifier(**param)
        elif method == 'SVM':
            model = SVC(**param)
        model.fit(X=x, y=y)
        x_predict = model.predict(x)
        score = (x_predict == y).sum() / len(y)
        self.x_predict = x_predict
        self.model = model
        self.score = score
        self.confusion_matrix = pd.DataFrame(confusion_matrix(y, x_predict), columns=model.classes_,
                                             index=model.classes_)
        self.classification_report = classification_report(y, x_predict)

    def model_predict(self, x=None):
        '''
        
        :param x: 待预测数据集
        :return: 预测结果集
        '''
        model = self.model
        if x is None:
            x = [self.x]
        predict_value = model.predict(x)
        return predict_value


if __name__ == '__main__':
    # 生成股价波动随机数
    delta = np.random.randn(10000)
    data = delta.cumsum() + 100
    plt.plot(data)
    plt.show()

    # 标签值
    y_delta = np.where(delta > 0, 'up', 'down')  # 判断涨跌
    y_delta_label = y_delta[100:]
    y = data[100:]  # 股价数据

    # 数据值，按照100天为一个周期，预测后一天的涨跌及股价
    x_delta = []
    for i in range(len(delta) - 100):
        x_delta.append(delta[i:i + 100])
    x_delta = np.array(x_delta)

    x = []
    for i in range(len(data) - 100):
        x.append(data[i:i + 100])
    x = np.array(x)

    # 拆分训练集和测试集
    x_train, y_train, x_test, y_test = train_test_split(x_delta, y_delta_label, test_size=0.3)

    # 创建模型
    model = stockmodel()
    # 训练模型,'Logistic'、'KNN'、'SVM'
    model.model_train(x=x_train, y=x_test, method='Logistic')
    # 训练集的准确率
    print('train score:', model.score)
    # 训练集的预测矩阵
    print('train confusion_matrix:\n', model.confusion_matrix)
    # 训练集的准确率矩阵
    print('train classification_report:\n', model.classification_report)

    # 预测
    y_test_predict = model.model_predict(x=y_train)
    print('predict limit 10:', y_test_predict[-10:])
    print('test score:', (y_test_predict == y_test).sum() / len(y_test))
