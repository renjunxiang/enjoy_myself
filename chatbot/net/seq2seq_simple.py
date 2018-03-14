from keras.models import Sequential
from keras.layers.recurrent import GRU
from keras.layers import Bidirectional
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dense, RepeatVector, Masking
from keras.layers.embeddings import Embedding
from keras.optimizers import SGD, Adagrad, Adam


def seq2seq_simple(input_dic_len=100,
                   input_len=50,
                   vector_len=200,
                   output_dim=100,
                   output_len=10):
    '''
    
    :param input_dic_len: 输入的字典长度
    :param input_len: 输入的文本长度
    :param vector_len: 词向量维度
    :param output_dim: 输出的字典长度
    :param output_len: 输出的文本长度
    :return: 
    '''

    # input_dic_len=100
    # input_len=50
    # vector_len=200
    # output_len=50
    # output_dic_len=100

    model = Sequential()
    # 创建词向量
    model.add(Embedding(input_dim=input_dic_len + 1,
                        input_length=input_len,
                        output_dim=vector_len,
                        mask_zero=0))
    # encoding过程
    model.add(Bidirectional(GRU(units=32,
                                activation='tanh',
                                recurrent_activation='hard_sigmoid',
                                return_sequences=False)))
    model.add(Dense(units=50,
                    activation='relu'))
    # decoding过程，按照输出长度复制encoding的结果
    model.add(RepeatVector(output_len))
    model.add(GRU(units=50, return_sequences=True, activation="relu"))
    model.add(TimeDistributed(Dense(units=output_dim, activation="relu")))
    optimizer = Adam(lr=0.01)
    model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
    return model
