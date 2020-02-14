# -*- coding: utf-8 -*-
"""
Changes on Feb 2020

@author: Lucas Rodrigues Cupertino Cardoso (lucas.cupertino@usp.br)

Original code by fangshuyang (yfs2016@hit.edu.cn) // Created on Feb 2018

"""

"""
## Pytorch
import torch.nn as nn
import torch
import torch.nn.functional as F
"""

## Keras' Implementation
import numpy as np
from numpy import random
from keras import layers
from keras import models
from keras import optimizers

## Kernel size = 3*3
## Stride = 1 // Padding = 1
## Activation = ReLU
## Initial data = (2000, 301, 29)
## After the first MaxPooling2D: (29, 401, 301)

## Random test
data_set_test = np.random.randn(401, 301)

model =  models.Sequential()

print('DNN implementation using Keras')

##class unetConv2(nn.Module):
class unetConv2(model):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(unetConv2, self).__init__()
        # Original sequence of operation if is_batchnorm = True:
        # Conv2D, BN, ReLU
        if is_batchnorm:
            self.conv1 = model.add(layers.Conv2D(in_size, (3, 3), strides = (1, 1), padding = 1, activation = 'relu'),
                                   layers.BatchNormalization(out_size))
            
            self.conv2 = model.add(layers.Conv2D(out_size, '''out_size,''' kernel_size = (3, 3), strides = (1, 1), padding = 1, activation = 'relu'),
                                   layers.BatchNormalization(out_size))

        else:
            self.conv1 = model.add(layers.Conv2d(in_size, '''out_size,''' kernel_size = (3, 3), strides = (1, 1), padding  = 1, activation = 'relu'))

            self.conv2 = model.add(layers.Conv2d(out_size, '''out_size,''' kernel_size = (3, 3), strides = (1, 1), padding = 1, activation = 'relu'))

                                       
    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


##class unetDown(nn.Module):
class unetDown(model): 
    def __init__(self, in_size, out_size, is_batchnorm):
        super(unetDown, self).__init__()
        self.conv = unetConv2(in_size, out_size, is_batchnorm)
        ##self.down = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.down = model.add(layers.MaxPooling2D(pool_size = (2,2)''', strides = 1'''))

    def forward(self, inputs):
        outputs = self.conv(inputs)
        outputs = self.down(outputs)
        return outputs


##class unetUp(nn.Module):
class unetUp(model):
    def __init__(self, in_size, out_size, is_deconv):
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size, out_size, True)
        # Transposed convolution
        if is_deconv:
            self.up = model.add(layers.Conv2DTranspose(in_size, kernel_size= (2, 2), stride = 2))
        else:
            self.up = model.add(layers.Upsampling2D(kernel_size = (2, 2), interpolation = 'bilinear'))

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset1 = (outputs2.size()[2]-inputs1.size()[2])
        offset2 = (outputs2.size()[3]-inputs1.size()[3])
        padding=[offset2//2,(offset2+1)//2,offset1//2,(offset1+1)//2]
        # Skip and concatenate 
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))


##class  UnetModel(nn.Module):
class UnetModel(model):
    def __init__(self, n_classes, in_channels ,is_deconv, is_batchnorm):
        super(UnetModel, self).__init__()
        self.is_deconv     = is_deconv
        self.in_channels   = in_channels
        self.is_batchnorm  = is_batchnorm
        self.n_classes     = n_classes
        
        filters = [64, 128, 256, 512, 1024, 2048]
        
        self.down1   = unetDown(self.in_channels, filters[0], self.is_batchnorm)
        self.down2   = unetDown(filters[0], filters[1], self.is_batchnorm)
        self.down3   = unetDown(filters[1], filters[2], self.is_batchnorm)
        self.down4   = unetDown(filters[2], filters[3], self.is_batchnorm)
        self.center  = unetConv2(filters[3], filters[4], self.is_batchnorm)
        self.up4     = unetUp(filters[4], filters[3], self.is_deconv)
        self.up3     = unetUp(filters[3], filters[2], self.is_deconv)
        self.up2     = unetUp(filters[2], filters[1], self.is_deconv)
        self.up1     = unetUp(filters[1], filters[0], self.is_deconv)
        self.final   = model.add(layers.Conv2D(filters[0],self.n_classes, 1))
        
    def forward(self, inputs,label_dsp_dim):
        down1  = self.down1(inputs)
        down2  = self.down2(down1)
        down3  = self.down3(down2)
        down4  = self.down4(down3)
        center = self.center(down4)
        up4    = self.up4(down4, center)
        up3    = self.up3(down3, up4)
        up2    = self.up2(down2, up3)
        up1    = self.up1(down1, up2)
        up1    = up1[:,:,1:1+label_dsp_dim[0],1:1+label_dsp_dim[1]].contiguous()
        
        return self.final(up1)
    
    # Initialization of Parameters
    def  _initialize_weights(self):
          for m in self.modules():
            if isinstance(m, keras.layers.Conv2D()):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, keras.layers.BatchNormalization()):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, keras.layers.Conv2DTranspose()):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()


## Showing the current model and it's layers
model.summary()
