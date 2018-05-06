from keras.models import Model
from keras.layers import Input, Embedding, GRU, Bidirectional, Dense, \
    RepeatVector, Masking, concatenate, Reshape, TimeDistributed
from keras.optimizers import SGD, Adagrad, Adam


def seq2seq_simple(input_dic_len=100,
                   input_len=50,
                   vector_len=200,
                   hidden_dim=100,
                   output_dim=100,
                   output_len=10):
    '''
    
    :param input_dic_len: 输入的字典长度
    :param input_len: 输入的文本长度
    :param vector_len: 词向量维度
    :param hidden_dim: encoding结尾的全连接节点数
    :param output_dim: 输出的字典长度
    :param output_len: 输出的文本长度
    :return: 
    '''

    # input_dic_len=100
    # input_len=50
    # vector_len=200
    # hidden_dim=100
    # output_dim = 100
    # output_len=50

    data_input = Input(shape=[input_len])
    # 创建词向量
    word_vec = Embedding(input_dim=input_dic_len + 1,
                         input_length=input_len,
                         output_dim=vector_len,
                         mask_zero=0,
                         name='Embedding')(data_input)
    # encoding过程
    rnn_encoding, state_h1, state_h2 = Bidirectional(GRU(units=32,
                                                         activation='tanh',
                                                         recurrent_activation='hard_sigmoid',
                                                         return_sequences=False,
                                                         return_state=True),
                                                     name='Bidirectional_encoding')(word_vec)
    data_encoding = Dense(units=hidden_dim,
                          activation='relu',
                          name='encoding')(rnn_encoding)
    # decoding过程，按照输出长度复制encoding的结果
    data_RepeatVector = RepeatVector(n=output_len)(data_encoding)
    # encoding过程的细胞状态作为decoding的初始值,增强信息传递
    rnn_decoding = Bidirectional(GRU(units=32,
                                     return_sequences=True,
                                     activation="relu"),
                                 name='Bidirectional_decoding')(data_RepeatVector, initial_state=[state_h1, state_h2])
    data_decoding = TimeDistributed(Dense(units=output_dim, activation="relu"),
                                    name='TimeDistributed')(rnn_decoding)
    optimizer = Adam(lr=0.01)
    model = Model(inputs=data_input, outputs=data_decoding)
    model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    from keras.utils import to_categorical
    import numpy as np
    import random

    input_len=5
    ask_transform = [[random.choice('abcdefg') for j in range(random.randint(1,input_len))]
                     for i in range(5000)]
    answer_transform = [[j.upper() for j in i] for i in ask_transform]

    tokenizer_ask = Tokenizer()
    tokenizer_ask.fit_on_texts(texts=ask_transform)
    ask_seq = tokenizer_ask.texts_to_sequences(texts=ask_transform)
    ask_new = pad_sequences(ask_seq, maxlen=input_len, padding='post', value=0, dtype='int')

    output_len = 5
    tokenizer_answer = Tokenizer()
    tokenizer_answer.fit_on_texts(texts=answer_transform)
    answer_seq = tokenizer_answer.texts_to_sequences(texts=answer_transform)
    answer_new = pad_sequences(answer_seq, maxlen=output_len, padding='post', value=0, dtype='int')
    answer_categorical = to_categorical(answer_new)

    model_seq2seq = seq2seq_simple(input_dic_len=len(tokenizer_ask.word_index),
                                   input_len=input_len,
                                   vector_len=20,
                                   hidden_dim=20,
                                   output_dim=answer_categorical.shape[2],
                                   output_len=output_len)

    model_seq2seq.fit(x=ask_new, y=answer_categorical, batch_size=50, epochs=10, validation_split=0.2,verbose=2)

    answer_key = list(tokenizer_answer.word_index.keys())
    answer_values = list(tokenizer_answer.word_index.values())

    def chatbot(text=None):
        text=tokenizer_ask.texts_to_sequences(texts=[text])
        text_new = pad_sequences(text, maxlen=input_len, padding='post', value=0, dtype='float32')
        result = model_seq2seq.predict(text_new)[0]
        result = [np.argmax(i) for i in result]
        result = [answer_key[answer_values.index(i)] for i in result if i in answer_values]
        return result


    for i in ask_transform[0:20]:
        print('ask:', i,'answer:', chatbot(text=i))


chatbot(['a','d','g','e','c'])