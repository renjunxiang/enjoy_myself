import pandas as pd
import os
import numpy as np
import time
import chardet

# DIR = os.path.dirname(__file__)

DIR = 'D:\\github\\enjoy_myself\\crawler'
data_names = os.listdir(DIR + '/data')

#load data
writer = pd.ExcelWriter(path=DIR + '/新奇书网.xlsx')
for data_name in data_names:
    try:
        data = np.load(DIR + '/data/' + data_name)
        classify = data_name.split('.npy')[0]
        data = pd.DataFrame(data,
                            columns=['title', 'url', 'author', 'size',
                                     'update', 'describtions', 'stars', 'classify'])
        try:
            data.to_excel(excel_writer=writer, sheet_name=classify, index=False, encoding='utf-8')
            print(classify, ' to excel direct')
        except:
            # 不能直接写进表明有内容存在编码问题，逐句查询
            print(classify, ' to excel direct failed')
            for i in range(len(data)):
                if i+1 % 100 == 0:
                    print('%d/%d'%(i,len(data)))
                try:
                    data.iloc[i:i + 1, 5:6].to_excel(DIR + '/try.xlsx')
                except:
                    # 不能直接写进的对每个词判断编码问题，目前发现的问题编码可以同时被utf-8和gbk解码
                    describtion_1 = ''
                    for word in data['describtions'][i]:
                        try:
                            word.encode('utf-8').decode('gbk')
                        except:
                            describtion_1 += word
                    data['describtions'][i] = describtion_1
            data.to_excel(excel_writer=writer, sheet_name=classify, index=False, encoding='utf-8')
            print(classify, ' to excel')
    except:
        print(classify, ' failed')
writer.close()
