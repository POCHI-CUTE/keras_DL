from __future__ import print_function
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.utils import np_utils
from make_tensorboad import make_tensorboard

np.random.seed(1671)

nb_epoch = 20
batch_size = 128
verbose = 1
nb_calsses = 10
optimizer = SGD()
n_hidden = 128
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

y_train = np_utils.to_categorical(y_train, nb_calsses)
y_test  = np_utils.to_categorical(y_test, nb_calsses)

#モデルの作成
model = Sequential()
model.add(Dense(n_hidden, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(n_hidden))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(nb_calsses))
model.add(Activation('softmax'))
model.summary()

callbacks = [make_tensorboard(set_dir_name='keras_MNIST_V3')]

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


model.fit(x_train,y_train,
batch_size=batch_size,
epochs=nb_epoch,
callbacks=callbacks,
verbose=verbose,
validation_split=validation_split)

score = model.evaluate(x_test, y_test, verbose=verbose)
print("\ntest score:", score[0])
print("test accuracy:", score[1])