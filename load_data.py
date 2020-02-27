#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 14:16:28 2020

@author: jonas
"""
import numpy as np
#import tensorflow as tf

from model import create_model


data = np.random.random((200,400,304,29))
data = tf.convert_to_tensor(data,dtype=tf.float32)


linhas=data.shape[1]
colunas= data.shape[2]
canais = data.shape[3]


create_model(linhas,colunas,canais)
