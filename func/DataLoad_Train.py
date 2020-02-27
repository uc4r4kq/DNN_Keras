# -*- coding: utf-8 -*-
"""
Load training data set
@author: jonas mendonca (jonas.mendonca@usp.br)"""

import numpy as np
from skimage.measure import block_reduce
import skimage
import scipy.io
from scipy import stats



def DataLoad_Train(train_size,train_data_dir,data_dim,in_channels,model_dim,data_dsp_blk,label_dsp_blk,start,datafilename,dataname,truthfilename,truthname,folder_dataset):
    for i in range(start,start+train_size):
        
        filename_seis = train_data_dir+folder_dataset[0]+datafilename+str(i)+'.mat'
        print (datafilename+str(i))
        data1_set = scipy.io.loadmat(filename_seis)
        data = data1_set[dataname]
        
        for j in range(0,29):
            vet_aux = np.zeros((2000,304,29))
            aux = data[:,:,j]
            aux2 = aux[:,297:300]
            aux3 = np.concatenate((aux,aux2),axis=1)
            vet_aux[:,:,j] = aux3

        data = vet_aux

        
        
        
        data1_set = np.float32(data.reshape([data_dim[0],data_dim[1],in_channels]))
         # Change the dimention [h, w, c] --> [c, h, w]
        for k in range (0,in_channels):
            data11_set     = np.float32(data1_set[:,:,k])
            data11_set     = np.float32(data11_set)
            # Data downsampling
            data11_set     = block_reduce(data11_set,block_size=data_dsp_blk,func=np.max)#decimate)
            data_dsp_dim   = data11_set.shape
            data11_set     = data11_set.reshape(1,data_dsp_dim[0]*data_dsp_dim[1])
            if k==0:
                train1_set = data11_set
            else:
                train1_set = np.append(train1_set,data11_set,axis=0)
        filename_label     = train_data_dir+folder_dataset[1]+truthfilename+str(i)
        print (truthfilename+str(i))
        data2_set          = scipy.io.loadmat(filename_label)
        data2_set          = np.float32(data2_set[str(truthname)].reshape(model_dim))
        # Label downsampling
        data2_set          = block_reduce(data2_set,block_size=label_dsp_blk,func=np.max)
        label_dsp_dim      = data2_set.shape
        data2_set          = data2_set.reshape(1,label_dsp_dim[0]*label_dsp_dim[1])
        data2_set          = np.float32(data2_set)
        if i==start:
            train_set      = train1_set
            label_set      = data2_set
        else:
            train_set      = np.append(train_set,train1_set,axis=0)
            label_set      = np.append(label_set,data2_set,axis=0)
    
    print ('Train_set:',train_set.shape, 'data_dsp dim:',data_dsp_dim[0],data_dsp_dim[1], 'train_size:',train_size)
            
    
    train_set = train_set.reshape((train_size,in_channels,data_dsp_dim[0]*data_dsp_dim[1]))
    label_set = label_set.reshape((train_size,1,label_dsp_dim[0]*label_dsp_dim[1]))
    
    return train_set, label_set, data_dsp_dim, label_dsp_dim

def decimate(a,axis):

    idx = np.round((a.shape[axis]+1.0)/2.0-1)
    return a[...,idx.astype(int)]


