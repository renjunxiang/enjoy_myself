from keras.models import Sequential
from keras.layers.recurrent import GRU
from keras.layers import Bidirectional
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dense, RepeatVector, Masking
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import SGD, Adagrad, Adam
from keras.utils import to_categorical
from keras import backend as K
import numpy as np
import pandas as pd
from keras import losses
import tensorflow as tf

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import jieba

train_data = pd.read_excel('./train_data.xlsx', sheet_name=0)

# with open('./stopwords.txt', 'r', encoding='utf-8') as f:
#     stopwords = f.read().splitlines()
ask = train_data['ask']
answer = train_data['answer']

data = [
    ['寥落古行宫', '宫花寂寞红'],
    ['白头宫女在', '闲坐说玄宗'],
    ['白日依山尽', '黄河入海流'],
    ['欲穷千里目', '更上一层楼'],
    ['三日入厨下', '洗手作羹汤'],
    ['未谙姑食性', '先遣小姑尝'],
    ['红豆生南国', '春来发几枝'],
    ['愿君多采撷', '此物最相思'],
    ['终南阴岭秀', '积雪浮云端'],
    ['林表明霁色', '城中增暮寒'],
    ['鸣筝金粟柱', '素手玉房前'],
    ['欲得周郎顾', '此物最相思'],
]

# ask = ['你好', '再见', '去死', '滚', '拜拜', '你好', '拜拜，拜拜']
# answer = ['你好，你好', '再见，再见', '去死，去死', '滚，滚', '拜拜', '你好', '拜拜']

ask=[]
answer=[]
for i in data:
    ask.append(i[0])
    answer.append(i[1])

def texts_transform(texts=None):
    texts_new = [' '.join(jieba.lcut(i)) for i in texts]
    return texts_new


ask_transform = texts_transform(texts=ask)
answer_transform = texts_transform(texts=answer)

tokenizer_ask = Tokenizer()
tokenizer_ask.fit_on_texts(texts=ask_transform)
ask_seq = tokenizer_ask.texts_to_sequences(texts=ask_transform)
ask_new = pad_sequences(ask_seq, maxlen=10, padding='post', value=0, dtype='int')

output_len = 10

tokenizer_answer = Tokenizer(num_words=50)
tokenizer_answer.fit_on_texts(texts=answer_transform)
answer_seq = tokenizer_answer.texts_to_sequences(texts=answer_transform)
answer_new = pad_sequences(answer_seq, maxlen=output_len, padding='post', value=0, dtype='int')
answer_categorical = to_categorical(answer_new)


# n1=[len(i) for i in ask_seq]
# n2=[len(i) for i in answer_seq]
#
# def myloss(y_true, y_pred):
#     w = tf.ones(y_pred.shape, dtype=tf.float32)
#     loss = tf.contrib.seq2seq.sequence_loss(logits=y_pred, targets=y_true, weights=w)
#     loss=tf.square(y_true-y_pred)
#     return loss

def seq2seq(input_dic_len=100,
            input_len=50,
            vector_len=200,
            output_dic_len=100):
    '''
    :param input_dim: 字典长度，即onehot的长度
    :param input_length: 文本长度
    :param output_dim: 词向量长度
    :return: 
    '''
    '''
    input_dic_len=100
            input_len=50
            vector_len=200
            output_len=50
            output_dic_len=100
    '''
    model = Sequential()
    model.add(Embedding(input_dim=input_dic_len + 1,
                        input_length=input_len,
                        output_dim=vector_len,
                        mask_zero=0))
    model.add(Masking(mask_value=0))
    model.add(Bidirectional(GRU(units=32,
                                activation='tanh',
                                recurrent_activation='hard_sigmoid',
                                return_sequences=False)))
    model.add(Dense(units=50,
                    activation='relu'))
    model.add(RepeatVector(output_len))
    model.add(GRU(units=50, return_sequences=True, activation="relu"))
    model.add(TimeDistributed(Dense(units=answer_categorical.shape[2], activation="relu")))
    optimizer = Adagrad(lr=0.01)
    model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
    return model


model_seq2seq = seq2seq(input_dic_len=len(tokenizer_ask.word_index),
                        input_len=10, vector_len=100)

model_seq2seq.fit(x=ask_new, y=answer_categorical, batch_size=25, epochs=10)

answer_key = list(tokenizer_answer.word_index.keys())
answer_values = list(tokenizer_answer.word_index.values())


def robot(text=None):
    text = [jieba.lcut(text)]
    text_seq = tokenizer_ask.texts_to_sequences(texts=text)
    text_new = pad_sequences(text_seq, maxlen=10, padding='post', value=0, dtype='float32')
    result = model_seq2seq.predict(text_new)[0]
    result = [np.argmax(i) for i in result]
    # result=np.random.randint(1,500,np.random.randint(10,50,1)[0])
    result = ''.join([answer_key[answer_values.index(i)] for i in result if i in answer_values])
    return result


for i in ask:
    print(i)
    print(robot(text=i))

model_seq2seq.predict(np.array([np.random.randint(1, 50, 10)]))

answer_key[answer_values.index(5)]
