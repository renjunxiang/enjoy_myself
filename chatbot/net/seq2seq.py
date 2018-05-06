from keras.models import Model
from keras.layers import Input, Embedding, GRU, Bidirectional, Dense, \
    RepeatVector, Masking, concatenate, Reshape, TimeDistributed
from keras.optimizers import SGD, Adagrad, Adam


def seq2seq(input_dic_len=10,
            input_len=10,
            vector_len=20,
            hidden_dim=32,
            output_dim=10,
            output_len=10):
    '''
    [X1,X2,...,Xm]->cell+hide+output->[Y1+cell1+hide1=RNN(cell+h,o),Y2=RNN(cell1+hide1,Y1),...,Yn=RNN(cell n-1 + hide n-1,Yn-1)]    
    :param input_dic_len: 输入的字典长度
    :param input_len: 输入的文本长度
    :param vector_len: 词向量维度
    :param hidden_dim: encoding结尾的全连接节点数和rnn核数量
    :param output_dim: 输出的字典长度
    :param output_len: 输出的文本长度
    :return: 
    '''

    # input_dic_len=10
    # input_len=10
    # vector_len=20
    # hidden_dim=32
    # output_dim = 10
    # output_len=10

    data_input = Input(shape=[input_len])
    # 创建词向量
    word_vec = Embedding(input_dim=input_dic_len + 1,
                         input_length=input_len,
                         output_dim=vector_len,
                         mask_zero=0,
                         name='Embedding')(data_input)
    # encoding过程
    rnn_encoding, state_h = GRU(units=hidden_dim,
                                activation='tanh',
                                recurrent_activation='hard_sigmoid',
                                return_sequences=False,
                                return_state=True)(word_vec)
    data_encoding = Dense(units=hidden_dim,
                          activation='relu',
                          name='encoding')(rnn_encoding)
    # decoding过程
    # encoding的状态作为decoding的初始，后续输出作为下一个输入
    initial_state = state_h
    decoding_input = RepeatVector(n=1)(data_encoding)
    data_decoding = []
    rnn_decoding = GRU(units=hidden_dim,
                       return_sequences=False,
                       return_state=True,
                       activation="relu",
                       name='decoding')
    for i in range(output_len):
        decoding_output, state_h = rnn_decoding(decoding_input, initial_state=initial_state)
        data_decoding.append(decoding_output)
        initial_state = state_h
        decoding_input = RepeatVector(n=1)(decoding_output)
    rnn_decoding = concatenate(inputs=data_decoding, axis=-1)
    rnn_decoding = Reshape(target_shape=[output_len, hidden_dim])(rnn_decoding)
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

    input_len = 8
    ask_transform = [[random.choice('abcdefghijklmnopqrstuvwxyz') for j in range(random.randint(3, input_len))]
                     for i in range(5000)]
    answer_transform = [[j.upper() for j in i] for i in ask_transform]
    '''
    ask_transform = [[random.choice('0123456789') for j in range(random.randint(3, input_len))]
                     for i in range(5000)]
    v = {'0': '零', '1': '一', '2': '二', '3': '三', '4': '四',
         '5': '五', '6': '六', '7': '七', '8': '八', '9': '九'}
    answer_transform = [[v[j] for j in i] for i in ask_transform]

    '''

    tokenizer_ask = Tokenizer()
    tokenizer_ask.fit_on_texts(texts=ask_transform)
    ask_seq = tokenizer_ask.texts_to_sequences(texts=ask_transform)
    ask_new = pad_sequences(ask_seq, maxlen=input_len, padding='post', value=0, dtype='int')

    output_len = 8
    tokenizer_answer = Tokenizer()
    tokenizer_answer.fit_on_texts(texts=answer_transform)
    answer_seq = tokenizer_answer.texts_to_sequences(texts=answer_transform)
    answer_new = pad_sequences(answer_seq, maxlen=output_len, padding='post', value=0, dtype='int')
    answer_categorical = to_categorical(answer_new)

    model_seq2seq = seq2seq(input_dic_len=len(tokenizer_ask.word_index),
                            input_len=input_len,
                            vector_len=20,
                            hidden_dim=50,
                            output_dim=answer_categorical.shape[2],
                            output_len=output_len)

    model_seq2seq.fit(x=ask_new, y=answer_categorical, batch_size=50, epochs=20, validation_split=0.2, verbose=2)

    answer_key = list(tokenizer_answer.word_index.keys())
    answer_values = list(tokenizer_answer.word_index.values())


    def chatbot(text=None):
        text = tokenizer_ask.texts_to_sequences(texts=[text])
        text_new = pad_sequences(text, maxlen=input_len, padding='post', value=0, dtype='float32')
        result = model_seq2seq.predict(text_new)[0]
        result = [np.argmax(i) for i in result]
        result = [answer_key[answer_values.index(i)] for i in result if i in answer_values]
        return result


    for i in ask_transform[0:20]:
        print('ask:', i, 'answer:', chatbot(text=i))
