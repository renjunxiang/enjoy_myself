from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
import pandas as pd
from net.seq2seq import seq2seq

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# train_data = pd.read_excel('./train_data.xlsx', sheet_name=0)
#
# # with open('./stopwords.txt', 'r', encoding='utf-8') as f:
# #     stopwords = f.read().splitlines()
# ask = train_data['ask']
# answer = train_data['answer']


# def reverse_num(x='123456'):
#     y = [x[len(x) - 1 - i] for i in range(len(x))]
#     return ' '.join(y)


alpha = 'qwertyuiopasdfghjklzxcvbnm'


def creat_word():
    n = np.random.randint(0, 26, np.random.randint(1, 15))
    return ' '.join([alpha[i] for i in n])


ask_maxlen = 20

ask_transform = [creat_word() for i in range(50000)]
answer_transform = [i.upper() for i in ask_transform]

tokenizer_ask = Tokenizer(num_words=None)
tokenizer_ask.fit_on_texts(texts=ask_transform)
ask_seq = tokenizer_ask.texts_to_sequences(texts=ask_transform)
ask_new = pad_sequences(ask_seq, maxlen=ask_maxlen, padding='post', value=0, dtype='int')

output_len = 20

tokenizer_answer = Tokenizer(num_words=50, lower=False)
tokenizer_answer.fit_on_texts(texts=answer_transform)
answer_seq = tokenizer_answer.texts_to_sequences(texts=answer_transform)
answer_new = pad_sequences(answer_seq, maxlen=output_len, padding='post', value=0, dtype='int')
answer_categorical = to_categorical(answer_new)

model_seq2seq = seq2seq(input_dic_len=len(tokenizer_ask.word_index),
                        input_len=ask_maxlen,
                        vector_len=20,
                        hidden_dim=50,
                        output_dim=answer_categorical.shape[2],
                        output_len=output_len)

model_seq2seq.fit(x=ask_new, y=answer_categorical, batch_size=128, epochs=20, validation_split=0.2)
model_seq2seq.save('./demo/models/seq2seq_epoch_20.h5')#准确率0.76

answer_key = list(tokenizer_answer.word_index.keys())
answer_values = list(tokenizer_answer.word_index.values())


# def robot(text=None):
#     text = [jieba.lcut(text)]
#     text_seq = tokenizer_ask.texts_to_sequences(texts=text)
#     text_new = pad_sequences(text_seq, maxlen=ask_maxlen, padding='post', value=0, dtype='float32')
#     result = model_seq2seq_simple.predict(text_new)[0]
#     result = [np.argmax(i) for i in result]
#     # result=np.random.randint(1,500,np.random.randint(10,50,1)[0])
#     result = ''.join([answer_key[answer_values.index(i)] for i in result if i in answer_values])
#     return result


def alpha_robot(text=None):
    text_seq = tokenizer_ask.texts_to_sequences(texts=text)
    text_new = pad_sequences(text_seq, maxlen=ask_maxlen, padding='post', value=0, dtype='float32')
    result = model_seq2seq.predict(text_new)[0]
    result = [np.argmax(i) for i in result]
    # result=np.random.randint(1,500,np.random.randint(10,50,1)[0])
    result = ''.join([answer_key[answer_values.index(i)] for i in result if i in answer_values])
    return result


for i in ask_transform[30:50]:
    print('ask:', ''.join(i.split(' ')))
    print('answer:', alpha_robot(text=[i]))
