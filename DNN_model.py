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
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, Conv2DTranspose

data = np.random.random((1,400,301,3))
linhas=data.shape[1]
colunas= data.shape[2]
canais = data.shape[3]

def reduzir(model,out_size,is_maxPooling):
    #Aplicando primeira redução
    model.add(Conv2D(filters=out_size, kernel_size=(3, 3), input_shape=(linhas, colunas, canais), strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    #Reaplicando no mesmo dado
    model.add(Conv2D(filters=out_size, kernel_size=(3, 3), input_shape=(linhas, colunas, canais), strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    if is_maxPooling: # na última camada não é aplicado o maxpooling, pois agora iremos reconstruir as features
        model.add(MaxPooling2D(pool_size=(2,2),padding='same'))

    return model


def aumentar(model,out_size,is_maxPooling):
    model.add(Conv2D(filters=out_size, kernel_size=(3, 3), input_shape=(linhas, colunas, canais), strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(filters=out_size, kernel_size=(3, 3), input_shape=(linhas, colunas, canais), strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    if is_maxPooling:
        model.add(Conv2DTranspose(filters=out_size,kernel_size=(2,2),strides=2,padding='same'))    #(pool_size=(2,2),padding='same'))

    return model





def create_model():
    model= Sequential()
    filters=[64,128,256,512,1024]
    
    #adicionei essa flag de True e False para saber quando não precisar mais fazer maxpooling
    reduzir(model,filters[0],True)
    reduzir(model,filters[1],True)
    reduzir(model,filters[2],True)
    reduzir(model,filters[3],True)
    reduzir(model,filters[4],False)
    
    aumentar(model,filters[3],True)    
    aumentar(model,filters[2],True)
    aumentar(model,filters[1],True)
    aumentar(model,filters[0],True)




    return model


def optimizer():
    return SGD(lr=1e-3)

model = create_model()
#model.compile(optmizer=optimizer())
model.summary()
