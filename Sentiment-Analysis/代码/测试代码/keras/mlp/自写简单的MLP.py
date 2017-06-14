#! /usr/bin/env python
#-*- coding: utf-8 -*-import numpy


from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.optimizers import SGD
from keras.utils import plot_model
from keras import backend as K
import theano

nb_classes=2
X_train=np.random.rand(20,20)
y_train=np.random.randint(2, size=20)

X_test=np.random.rand(20,20)
y_test=np.random.randint(2, size=20)
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

batch_size=16
nb_epoch=5

model = Sequential()
# Dense(input, output, init=’wegiths initial method’)
model.add(Dense(64, input_dim=20))
model.add(Activation('tanh')) # 激活函数
model.add(Dropout(0.5))     #采用50%的dropout

model.add(Dense(64, input_dim=64))
model.add(Activation('tanh'))
model.add(Dropout(0.5))

model.add(Dense(2, input_dim=64))
model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)#设定学习速度，衰减量等
model.compile(loss='mean_squared_error', optimizer=sgd) #损失函数为均方误差

# ………此处是加载训练数据的的代码。

# 开始训练。nb_epoch是迭代次数，batcn_size是数据块大小。


model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    epochs=nb_epoch, batch_size=batch_size)
score = model.evaluate(X_test, y_test,
                       batch_size=batch_size)


print('Test loss:', score[0])
print('Test accuracy:', score[1])
# print("history",history.history)


