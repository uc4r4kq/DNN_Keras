import numpy as np
from skimage.measure import block_reduce
import skimage
import scipy.io
from scipy import stats


def DataLoad_Test(train_size,train_data_dir,data_dim,in_channels,model_dim,data_dsp_blk,label_dsp_blk,start,datafilename,dataname,truthfilename,truthname,folder_dataset):
    train_size = 10;
    X_test = np.zeros((train_size, 400, 304, in_channels))
    Y_test = np.zeros((train_size, 202, 302,1))
    
    cont = 0
    for i in range(191,201):
        
        filename_seis = train_data_dir+folder_dataset[0]+datafilename+str(i)+'.mat'
        print (datafilename+str(i))
        data1_set = scipy.io.loadmat(filename_seis)
        data = data1_set[dataname]
        
        vet_aux = np.zeros((2000,304,29))
        for j in range(0,29):
            
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
            X_test[cont,:,:,k] = data11_set
        filename_label     = train_data_dir+folder_dataset[1]+truthfilename+str(i)
        print (truthfilename+str(i))
        data2_set          = scipy.io.loadmat(filename_label)
        data2_set          = np.float32(data2_set[str(truthname)].reshape(model_dim))
        # Label downsampling
        data2_set          = block_reduce(data2_set,block_size=label_dsp_blk,func=np.max)
        label_dsp_dim      = data2_set.shape
        #data2_set          = data2_set.reshape(1,label_dsp_dim[0]*label_dsp_dim[1])
        
        aux = data2_set
        aux2 = aux[:,299:300]
        data2_set = np.concatenate((aux,aux2),axis=1)
#        data2_set= np.float32(data2_set)
        
        aux = data2_set
        aux2 = aux[198:200,:]
        data2_set = np.concatenate((aux,aux2),axis=0)
        data2_set = data2_set[:-1,:]

        data2_set= np.float32(data2_set)
        
        data2_set = data2_set.reshape(202,302,1)
        Y_test[cont,:,:,:] = data2_set;
        cont = cont+1
    return X_test,Y_test      
