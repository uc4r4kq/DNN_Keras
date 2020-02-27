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
from keras.layers.merge import concatenate
#import torch

data = np.random.random((5,400,301,3))
data = tf.convert_to_tensor(data,dtype=tf.float32)
linhas=data.shape[1]
colunas= data.shape[2]
canais = data.shape[3]

def apply_maxPooling(camada):
    camada = MaxPooling2D(pool_size=(2,2),padding='same') (camada)
    return camada

def apply_deconvolution(camada,out_size):

    camada = Conv2DTranspose(filters=out_size,kernel_size=(2,2),strides=2,padding='same') (camada)
    return camada

def reduzir(camada,out_size):
    camada = Conv2D(filters=out_size, kernel_size=(3, 3), input_shape=(linhas, colunas, canais), strides=1, padding='same') (camada)
    camada = BatchNormalization() (camada)
    camada = Activation('relu') (camada)
    
    #reaplicando novamente no dado
    camada = Conv2D(filters=out_size, kernel_size=(3, 3), input_shape=(linhas, colunas, canais), strides=1, padding='same') (camada)
    camada = BatchNormalization() (camada)
    camada = Activation('relu') (camada)
    
    return camada

def aumentar(camada_corrente,camada_de_reducao_correspondente,out_size,remove_line):
   # camada = apply_deconvolution(camada_corrente,out_size)
    camada = Conv2DTranspose(filters=out_size,kernel_size=(2,2),strides=2,padding='same') (camada_corrente)
    camada = concatenate([camada,camada_de_reducao_correspondente],axis=3) #aqui concateno o dado após a deconvolução com o dado correspondente na camada de redução
    camada = reduzir(camada,out_size)

    return camada



def create_model():
    inputs = Input((linhas,colunas,canais))
    
    filters=[64,128,256,512,1024]
#    c1 = reduzir(inputs,filtros[0])
#    c2 = reduzir(apply_maxPooling(c1),filtros[1])
#    c3 = reduzir(apply_maxPooling(c2),filtros[2])
#    c4 = reduzir(apply_maxPooling(c3),filtros[3])
 #   c5 = reduzir(apply_maxPooling(c4),filtros[4])
    
    ######  C1 ##########
    out_size=filters[0]
    c1 = Conv2D(filters=out_size, kernel_size=(3, 3), input_shape=(linhas, colunas, canais), strides=1, padding='same') (inputs)
    c1 = BatchNormalization() (c1)
    c1 = Activation('relu') (c1)

    c1 = Conv2D(filters=out_size, kernel_size=(3, 3), input_shape=(linhas, colunas, canais), strides=1, padding='same') (c1)
    c1 = BatchNormalization() (c1)
    c1 = Activation('relu') (c1)
    
    c1_max = MaxPooling2D(pool_size=(2,2),padding='same') (c1)

    ####### C2 ################
    out_size=filters[1]
    c2 = Conv2D(filters=out_size, kernel_size=(3, 3), input_shape=(linhas, colunas, canais), strides=1, padding='same') (c1_max)
    c2 = BatchNormalization() (c2)
    c2 = Activation('relu') (c2)

    c2 = Conv2D(filters=out_size, kernel_size=(3, 3), input_shape=(linhas, colunas, canais), strides=1, padding='same') (c2)
    c2 = BatchNormalization() (c2)
    c2 = Activation('relu') (c2)
#    print (c2.shape)
    c2_max = MaxPooling2D(pool_size=(2,2),padding='same') (c2)
    
    ####### C3 ################
    out_size=filters[2]
    c3 = Conv2D(filters=out_size, kernel_size=(3, 3), input_shape=(linhas, colunas, canais), strides=1, padding='same') (c2_max)
    c3 = BatchNormalization() (c3)
    c3 = Activation('relu') (c3)

    c3 = Conv2D(filters=out_size, kernel_size=(3, 3), input_shape=(linhas, colunas, canais), strides=1, padding='same') (c3)
    c3 = BatchNormalization() (c3)
    c3 = Activation('relu') (c3)

    c3_max = MaxPooling2D(pool_size=(2,2),padding='same') (c3)
    
    
    ####### C4 ################
    out_size = filters[3]
    c4 = Conv2D(filters=out_size, kernel_size=(3, 3), input_shape=(linhas, colunas, canais), strides=1, padding='same') (c3_max)
    c4 = BatchNormalization() (c4)
    c4 = Activation('relu') (c4)

    c4 = Conv2D(filters=out_size, kernel_size=(3, 3), input_shape=(linhas, colunas, canais), strides=1, padding='same') (c4)
    c4 = BatchNormalization() (c4)
    c4 = Activation('relu') (c4)

    c4_max = MaxPooling2D(pool_size=(2,2),padding='same') (c4)

    ####### C5 ################
    out_size = filters[4]
    c5 = Conv2D(filters=out_size, kernel_size=(3, 3), input_shape=(linhas, colunas, canais), strides=1, padding='same') (c4_max)
    c5 = BatchNormalization() (c5)
    c5 = Activation('relu') (c5)

    c5 = Conv2D(filters=out_size, kernel_size=(3, 3), input_shape=(linhas, colunas, canais), strides=1, padding='same') (c5)
    c5 = BatchNormalization() (c5)
    c5 = Activation('relu') (c5)

    ######### C6 ##############
    out_size = filters[3]
    c6 = Conv2DTranspose(filters=out_size,kernel_size=(2,2),strides=2,padding='same') (c5) # Aplico a deconvolução
    c6 = concatenate([c6,c4],axis=3) #aqui concateno o dado após a deconvolução com o dado correspondente na camada de redução
    
    c6 = Conv2D(filters=out_size, kernel_size=(3, 3), input_shape=(linhas, colunas, canais), strides=1, padding='same') (c6)
    c6 = BatchNormalization() (c6)
    c6 = Activation('relu') (c6)

    c6 = Conv2D(filters=out_size, kernel_size=(3, 3), input_shape=(linhas, colunas, canais), strides=1, padding='same') (c6)
    c6 = BatchNormalization() (c6)
    c6 = Activation('relu') (c6)


    ######### C7 ##############
    out_size = filters[2]
    c7 = Conv2DTranspose(filters=out_size,kernel_size=(2,2),strides=2,padding='same') (c6) # Aplico a deconvolução
    c7 = concatenate([c7,c3],axis=3) #aqui concateno o dado após a deconvolução com o dado correspondente na camada de redução

    c7 = Conv2D(filters=out_size, kernel_size=(3, 3), input_shape=(linhas, colunas, canais), strides=1, padding='same') (c7)
    c7 = BatchNormalization() (c7)
    c7 = Activation('relu') (c7)

    c7 = Conv2D(filters=out_size, kernel_size=(3, 3), input_shape=(linhas, colunas, canais), strides=1, padding='same') (c7)
    c7 = BatchNormalization() (c7)
    c7 = Activation('relu') (c7)


    ######### C8 ##############
    out_size = filters[1]
    c8 = Conv2DTranspose(filters=out_size,kernel_size=(2,2),strides=2,padding='same') (c7) # Aplico a deconvolução
    c8 = Activation('relu') (c8)
    c2_aux = c2[:,:,:,-3:-1]
    c2_aux = c2[:,:,:,:-2]
    print (c8.shape)
    print (c2_aux.shape)
    c8 = concatenate([c8,c2],axis=3) #aqui concateno o dado após a deconvolução com o dado correspondente na camada de redução

    c8 = Conv2D(filters=out_size, kernel_size=(3, 3), input_shape=(linhas, colunas, canais), strides=1, padding='same') (c8)
    c8 = BatchNormalization() (c8)
    c8 = Activation('relu') (c8)

    c8 = Conv2D(filters=out_size, kernel_size=(3, 3), input_shape=(linhas, colunas, canais), strides=1, padding='same') (c8)
    c8 = BatchNormalization() (c8)
    c8 = Activation('relu') (c8)















#    c6 = Conv2DTranspose(filters=filtros[3],kernel_size=(2,2),strides=2,padding='same') (c5)
#    c6 = concatenate([c6,c4],axis=3) #aqui concateno o dado após a deconvolução com o dado correspondente na camada de redução
#    c6 = reduzir(c6,filtros[3])
    
#    c7 = Conv2DTranspose(filters=filtros[2],kernel_size=(2,2),strides=2,padding='same') (c6)
#    c7 = concatenate([c7,c3],axis=3) #aqui concateno o dado após a deconvolução com o dado correspondente na camada de redução
#    c7 = reduzir(c7,filtros[2])

#    c8 = Conv2DTranspose(filters=filtros[1],kernel_size=(2,2),strides=2,padding='same') (c7)
#    caux = c8[:,:,-1,:]
#    c8 = concatenate([c8[:,:,-1,:],c2],axis=3) #aqui concateno o dado após a deconvolução com o dado correspondente na camada de redução
#    c8 = reduzir(c8,filtros[1])

    #c6 = aumentar(c5,c4,filtros[3],False)
    #print (c6.shape)
    #c7 = aumentar(c6,c3,filtros[2],False)
    #print (c7.shape)
    #c8 = aumentar(c7,c2,filtros[1],True)
    #c9 = aumentar(c8,c1,filtros[0],False)
    
    

    model = Model(inputs=[inputs],outputs=[c7])
    model.summary()

create_model()
