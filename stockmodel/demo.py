import pandas as pd
from stockmodel import stockmodel
import os
from sklearn.model_selection import train_test_split

DIR = os.getcwd()
data = pd.read_excel(DIR + '/data.xlsx', sheet_name=0)
data = list(data.iloc[:, 0])

workbook_name = 'data_predict'
log_name = 'model_evaluate'
length = 40
# 创建模型
model = stockmodel()
# 转换数据
x, y = model.data_transform(data=data, type='delta', length=length)
# 拆分数据
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)

data_transform = pd.DataFrame(x, columns=['time_%d' % i for i in range(1, 1 + length)])
data_transform['label'] = y

# 写入excel
workbook = pd.ExcelWriter(path=DIR + '/%s.xlsx' % workbook_name)
raw = pd.DataFrame({'data': data + [None],
                    'label': [None] * length + list(y) + [None]},
                   columns=['data', 'label'])
f = open(DIR + '/%s.txt' % log_name, mode='w')
for model_name in ['Logistic', 'KNN', 'SVM']:
    # 训练集
    model.model_train(x=train_x, y=train_y, method=model_name)
    print('train score_%s' % model_name, model.score)
    f.write('train score_%s: %f\n' % (model_name, model.score))

    # 测试集
    y_test_predict = model.model_predict(x=test_x)

    # 评估测试集
    test_score = (y_test_predict == test_y).sum() / len(test_y)
    print('test score_%s' % model_name, test_score)
    f.write('test score_%s: %f\n' % (model_name, test_score))

    # 全部数据预测
    y_new = model.model_predict(x=None)
    y_predict = model.model_predict(x=x)
    data_transform[model_name] = y_predict

    # 拼接数据
    predict = [None] * length + list(y_predict) + list(y_new)
    raw['predict_%s' % model_name] = predict
    print('=======finish: %s======' % model_name)
    f.write('=======finish: %s======\n\n' % model_name)
f.close()
raw.to_excel(workbook, sheet_name='data_raw')
data_transform.to_excel(workbook, sheet_name='data_transform')
workbook.close()
