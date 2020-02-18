import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
# Loading the CIFAR-10 datasets
from keras.datasets import cifar10
batch_size = 32 
n_classes = 10 
epochs = 40
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
height = x_train.shape[1]
width = x_train.shape[2]# Validation dataset splitting
x_val = x_train[:5000,:,:,:]
y_val = y_train[:5000]
x_train = x_train[5000:,:,:,:]
y_train = y_train[5000:]
print('Training dataset: ', x_train.shape, y_train.shape)
print('Validation dataset: ', x_val.shape, y_val.shape)
print('Test dataset: ', x_test.shape, y_test.shape)

# Convert labels to categorical
y_train = np_utils.to_categorical(y_train, n_classes)
y_val = np_utils.to_categorical(y_val, n_classes)
y_test = np_utils.to_categorical(y_test, n_classes)# Datasets pre-processing
x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_val /= 255
x_test /= 255


def create_model():
  model = Sequential()
  model.add(Conv2D(filters=128, kernel_size=(3, 3), input_shape=(height, width, 3), strides=1, padding='same', activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=1, activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(0.25))
  model.add(Conv2D(filters=64, kernel_size=(2, 2), strides=1, activation='relu'))
  model.add(MaxPooling2D(pool_size=(1,1)))
  model.add(Conv2D(filters=32, kernel_size=(2, 2), strides=1, activation='relu'))
  model.add(MaxPooling2D(pool_size=(1,1)))
  model.add(Conv2D(filters=32, kernel_size=(2, 2), strides=1, activation='relu'))
  model.add(MaxPooling2D(pool_size=(1,1)))
  model.add(Dropout(0.25))
  model.add(Flatten())
  model.add(Dense(64, activation='relu'))
  model.add(Dense(n_classes, activation='softmax'))
  return model

def optimizer():
    return SGD(lr=1e-2)

model = create_model()
model.compile(optimizer=optimizer(),
              loss='categorical_crossentropy',
              metrics=['accuracy']
              )
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val,y_val),verbose=1)
model.summary()
