from __future__ import print_function
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import  Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
import os
from time import gmtime, strftime
from keras.callbacks import TensorBoard

#可視化
def make_tensorboard(set_dir_name=''):
    tictoc = strftime("%a_%d_%b_%Y_%H_%M_%S", gmtime())
    directory_name = tictoc
    log_dir =set_dir_name + '_' + directory_name
    os.mkdir(log_dir)
    tensorboard = TensorBoard(log_dir=log_dir)
    return tensorboard


#データの正規化
np.random.seed(1671)

NB_epoch = 200
batch_size = 128
verbose = 1
NB_calsses = 10
optimizer = SGD()
N_hidden = 128
validation_split=0.2



(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784)
x_test  = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test  = x_test.astype('float32')

x_train /= 255
x_test  /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = np_utils.to_categorical(y_train, NB_calsses)
y_test  = np_utils.to_categorical(y_test, NB_calsses)

#モデルの作成
model = Sequential()
model.add(Dense(NB_calsses, input_shape=(784,)))
model.add(Activation('softmax'))
model.summary()

callbacks = [make_tensorboard(set_dir_name='keras_MNIST_V1')]
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=NB_epoch, callbacks=callbacks,
verbose=verbose, validation_split=validation_split)

score = model.evaluate(x_test, y_test, verbose=verbose)
print("test score:", score[0])
print("test accuracy", score[1])