#import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import Model#Sequential, Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
#from keras.optimizers import SGD
from keras.layers.convolutional import MaxPooling2D, Conv2D
#from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, Conv2DTranspose, Input, concatenate, Cropping2D
from keras.layers.merge import concatenate
#import torch

#data = np.random.random((500,400,304,29))
#data = tf.convert_to_tensor(data,dtype=tf.float32)
#linhas=data.shape[1]
#colunas= data.shape[2]
#canais = data.shape[3]

from ParamConfig import *
from PathConfig import *
from LibConfig import *


train_set,label_set,data_dsp_dim,label_dsp_dim  = DataLoad_Train(train_size=TrainSize,train_data_dir=train_data_dir, \
                                                                 data_dim=DataDim,in_channels=Inchannels, \
                                                                 model_dim=ModelDim,data_dsp_blk=data_dsp_blk, \
                                                                 label_dsp_blk=label_dsp_blk,start=1, \
                                                                 datafilename=datafilename,dataname=dataname, \
                                                                 truthfilename=truthfilename,truthname=truthname, folder_dataset=folder_dataset)
# Change data type (numpy --> tensor)






def apply_maxPooling(camada):
    camada = MaxPooling2D(pool_size=(2,2),padding='same') (camada)
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


def reduzir_last(camada,out_size):
    camada = Conv2D(filters=out_size, kernel_size=(3, 3), input_shape=(linhas, colunas, canais), strides=1, padding='same') (camada)
    camada = BatchNormalization() (camada)
    camada = Activation('relu') (camada)

    return camada

def aumentar_last(camada_corrente,camada_de_reducao_correspondente,out_size):
    camada = Conv2DTranspose(filters=out_size,kernel_size=(2,2),strides=2,padding='same') (camada_corrente)
    camada = concatenate([camada,camada_de_reducao_correspondente],axis=3) #aqui concateno o dado após a deconvolução com o dado correspondente na camada de redução
    camada = reduzir_last(camada,out_size)

    return camada



def aumentar(camada_corrente,camada_de_reducao_correspondente,out_size):
    camada = Conv2DTranspose(filters=out_size,kernel_size=(2,2),strides=2,padding='same') (camada_corrente)
    camada = concatenate([camada,camada_de_reducao_correspondente],axis=3) #aqui concateno o dado após a deconvolução com o dado correspondente na camada de redução
    camada = reduzir(camada,out_size)

    return camada



def create_model(linhas,colunas,canais):
    inputs = Input((linhas,colunas,canais))
    
    filtros=[64,128,256,512,1024]
    c1 = reduzir(inputs,filtros[0])
    c2 = reduzir(apply_maxPooling(c1),filtros[1])
    c3 = reduzir(apply_maxPooling(c2),filtros[2])
    c4 = reduzir(apply_maxPooling(c3),filtros[3])
    c5 = reduzir(apply_maxPooling(c4),filtros[4])
    

    c6 = aumentar(c5,c4,filtros[3])
    c7 = aumentar(c6,c3,filtros[2])
    c8 = aumentar(c7,c2,filtros[1])
    c9 = aumentar_last(c8,c1,filtros[0])
    
    c9 = Cropping2D(cropping=(99, 1)) (c9) #fiz um crop na imagem que era (400,304) e deixei com (202,302)
    c9 = reduzir_last(c9,filtros[0])
    c9 = Conv2D(filters=1, kernel_size=(1, 1), strides=1) (c9)

    model = Model(inputs=[inputs],outputs=[c9])
    model.summary()

#create_model(linhas,colunas,canais)
