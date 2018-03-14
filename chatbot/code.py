from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
import pandas as pd
from net.seq2seq import seq2seq
import jieba
import os
from random import sample
from sklearn.model_selection import train_test_split


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def reverse_num(x='123456'):
    y = [x[len(x) - 1 - i] for i in range(len(x))]
    return ''.join(y)

ask_transform = [' '.join(list(np.random.randint(1, 10, np.random.randint(1, 31)).astype(str))) for i in range(1000)]
answer_transform = [reverse_num(i) for i in ask_transform]


# def texts_transform(texts=None):
#     texts_new = [' '.join(jieba.lcut(i)) for i in texts]
#     return texts_new
#
#
# ask_transform = texts_transform(texts=ask)
# answer_transform = texts_transform(texts=answer)

tokenizer_ask = Tokenizer()
tokenizer_ask.fit_on_texts(texts=ask_transform)
ask_seq = tokenizer_ask.texts_to_sequences(texts=ask_transform)
ask_new = pad_sequences(ask_seq, maxlen=30, padding='post', value=0, dtype='int')

output_len = 30

tokenizer_answer = Tokenizer()
tokenizer_answer.fit_on_texts(texts=answer_transform)
answer_seq = tokenizer_answer.texts_to_sequences(texts=answer_transform)
answer_new = pad_sequences(answer_seq, maxlen=output_len, padding='post', value=0, dtype='int')
answer_categorical = to_categorical(answer_new)

model_seq2seq = seq2seq(input_dic_len=len(tokenizer_ask.word_index),
                        input_len=30,
                        vector_len=50,
                        hidden_dim=20,
                        output_dim=answer_categorical.shape[2],
                        output_len=output_len)

# train_x, test_x, train_y, test_y = train_test_split(ask_new, answer_categorical, test_size=0.9)
model_seq2seq.fit(x=ask_new, y=answer_categorical, batch_size=50, epochs=5,validation_split=0.2)

answer_key = list(tokenizer_answer.word_index.keys())
answer_values = list(tokenizer_answer.word_index.values())


def robot(text=None):
    text = [jieba.lcut(text)]
    text_seq = tokenizer_ask.texts_to_sequences(texts=text)
    text_new = pad_sequences(text_seq, maxlen=30, padding='post', value=0, dtype='float32')
    result = model_seq2seq.predict(text_new)[0]
    result = [np.argmax(i) for i in result]
    # result=np.random.randint(1,500,np.random.randint(10,50,1)[0])
    result = ''.join([answer_key[answer_values.index(i)] for i in result if i in answer_values])
    return result

def num_robot(text=None):
    text_new = pad_sequences(ask_new[0], maxlen=30, padding='post', value=0, dtype='float32')
    result = model_seq2seq.predict(text_new)[0]
    result = [np.argmax(i) for i in result]
    # result=np.random.randint(1,500,np.random.randint(10,50,1)[0])
    result = ''.join([answer_key[answer_values.index(i)] for i in result if i in answer_values])
    return result

for i in ask_new[0:10]:
    print('ask:', i)
    print('answer:', num_robot(text=i))

111

model_seq2seq.predict(np.array([np.random.randint(1, 50, 10)]))

answer_key[answer_values.index(5)]
