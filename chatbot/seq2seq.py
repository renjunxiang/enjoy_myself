from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dense, RepeatVector, Masking
from keras.preprocessing.text import one_hot, hashing_trick, Tokenizer,text_to_word_sequence

import jieba

texts_x=['你好','吃过饭了吗','洗澡了吗']
texts_y=['你好','吃了','洗了']
jieba.lcut('我爱北京天安门')


one_hot(text='我 爱 北京 天安门', split=' ', n=100000)
hashing_trick(text='我 爱 北京 天安门', split=' ', n=10)

text = ['我', '爱', '北京']

tokenizer = Tokenizer(num_words=20, split=' ')
tokenizer.fit_on_texts(['我 爱 北京 天安门', '我 爱 苹果'])

tokenizer.texts_to_sequences(texts='我 爱 北京 天安门')
tokenizer.texts_to_matrix(texts=['我 爱 北京 天安门', '我 爱 苹果'], mode='binary')
tokenizer.sequences_to_matrix([a])
tokenizer.w

text_to_word_sequence(text='我 爱 北京 天安门')

x = [
    [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]],
    [[0.2, 0.3, 0.4], [0.3, 0.4, 0.5]],
]

y = [
    [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.2, 0.3, 0.4], [0.2, 0.3, 0.4]],
    [[0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [0.3, 0.4, 0.5], [0.2, 0.3, 0.4]],
]

input_shape = [2, 3]
output_shape = [4, 3]

model = Sequential()
# model.add(Masking(mask_value=0, input_shape=[10,20]))
# Encoder(第一个 LSTM)
model.add(LSTM(input_shape=[2, 3], output_dim=5, return_sequences=False))
model.add(Dense(output_dim=10, activation="relu"))
# 使用 "RepeatVector" 将 Encoder 的输出(最后一个 time step)复制 N 份作为 Decoder 的 N 次输入
model.add(RepeatVector(output_shape[0]))
# Decoder(第二个 LSTM)
model.add(LSTM(output_dim=5, return_sequences=True))
# TimeDistributed 是为了保证 Dense 和 Decoder 之间的一致
model.add(TimeDistributed(Dense(output_dim=output_shape[1], activation="linear")))
model.compile(loss="mse", optimizer='adam')
model.fit(x=x, y=y, epochs=1)


model.predict(x=x)