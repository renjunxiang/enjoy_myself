from keras.models import Model
from keras.layers import Input, Embedding, GRU, Bidirectional, Dense, \
    RepeatVector, Masking, concatenate, Reshape, TimeDistributed
from keras.optimizers import SGD, Adagrad, Adam
import numpy as np


def seq2seq(input_dic_len=100,
            input_len=50,
            vector_len=200,
            hidden_dim=20,
            output_dim=100,
            output_len=20):
    '''
    [X1,X2,...,Xm]->c->[Y1=RNN(c,c),Y2=RNN(c,Y1),...,Yn=RNN(c,Yn-1)]
    
    :param input_dic_len: 输入的字典长度
    :param input_len: 输入的文本长度
    :param vector_len: 词向量维度
    :param hidden_dim=20:encoding维数
    :param output_dic_len: 输出的字典长度+1
    :param output_len: 输出的文本长度
    :return: 
    x=np.array([[1,2],[2,3]])
    y=np.array([[0,0,1],[0,1,0]])
    '''

    # input_dic_len=3
    # input_len=2
    # vector_len=5
    # hidden_dim = 5
    # output_dim=3
    # output_len=2

    input_data = Input(shape=[input_len])
    # 创建词向量
    x = Embedding(input_dim=input_dic_len + 1,
                  input_length=input_len,
                  output_dim=vector_len,
                  mask_zero=0,
                  name='Embedding')(input_data)
    # encoding过程
    x = Bidirectional(GRU(units=32,
                          activation='tanh',
                          recurrent_activation='hard_sigmoid',
                          return_sequences=False),
                      name='Bidirectional_GRU')(x)
    encoding = Dense(units=hidden_dim,
                     activation='relu',
                     name='encoding')(x)
    # t-1时刻的输出，作为t时刻的输入，为了提高准确率每个时刻都将encoding的结果加进去
    ti = concatenate(inputs=[encoding, encoding], axis=-1, name='decoding_input')
    decoding_all = []
    for i in range(output_len):
        print('creat decoding layer: %d'%i)
        ti = Reshape(target_shape=[1, hidden_dim * 2])(ti)
        decoding_t = GRU(units=hidden_dim,
                         return_sequences=False,
                         activation="relu",
                         name='gru_decoding_%d' % i)(ti)
        decoding_all.append(decoding_t)
        ti = concatenate(inputs=[encoding, decoding_t], axis=1, name='decoding_time_%d' % i)
    decoding_concat = concatenate(inputs=decoding_all, axis=-1)
    decoding_concat = Reshape(target_shape=[output_len, hidden_dim])(decoding_concat)
    out_puts = TimeDistributed(Dense(units=output_dim, activation="relu",
                                     name='decoding_all'))(decoding_concat)
    model = Model(inputs=input_data, outputs=out_puts)

    optimizer = Adagrad(lr=0.01)
    model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
    return model

