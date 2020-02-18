import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.layers.convolutional import MaxPooling2D, Conv2D
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, Conv2DTranspose, Input
#import torch

data = np.random.random((400,301,3))
data = tf.convert_to_tensor(data,dtype=tf.float32)
print (data.shape)
linhas=data.shape[0]
colunas= data.shape[1]
canais = data.shape[2]




out_size=16
inputs = Input((linhas,colunas,canais))
c1 = Conv2D(filters=out_size, kernel_size=(3, 3), input_shape=(linhas, colunas, canais), strides=1, padding='same') (inputs)
c1 = BatchNormalization() (c1)
ac1 = Activation('relu') (c1)
    
model = Model(inputs=[inputs], outputs=[c1])
model.summary()


