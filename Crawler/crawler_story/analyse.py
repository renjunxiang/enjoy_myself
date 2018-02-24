import pandas as pd
import os
import numpy as np

# DIR = os.path.dirname(__file__)

DIR = 'D:\\github\\enjoy_myself\\crawler\\crawler_story'
data = np.load(DIR + '/data/' + '女频言情.npy')
data = pd.DataFrame(data,
                    columns=['title', 'url', 'author', 'size',
                             'update', 'describtions', 'stars', 'classify'])

from TextClustering.TextClustering import TextClustering

model = TextClustering(texts=list(data.iloc[:, 5]))
model.text_cut(stopwords_path='default')
model.creat_vocab(size=50, window=5, vocab_savepath=DIR + '/model/size50_window5.model')
# model.word2matrix(method='frequency', top=50)
model.word2matrix(method='vector', top=20, similar_n=10)
model.decomposition()
print('前两个成分的特征占比:', model.pca.explained_variance_ratio_[0:2].sum())
model.clustering(X=model.decomposition_data, model_name='KMeans', n_clusters=10)
model.show_decomposition(background=True, show=True, pixel=None, textsize=50)


