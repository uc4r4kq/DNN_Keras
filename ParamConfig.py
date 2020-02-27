# -*- coding: utf-8 -*-
"""
Parameters setting
@author: Jonas Mendonca (jonas.mendonca@usp.br)
"""


####################################################
####             MAIN PARAMETERS                ####
####################################################
SimulateData  = True          # If False denotes training the CNN with SEGSaltData
ReUse         = False#True         # If False always re-train a network 
use_seed      = True          # if we want use seed in weight in DNN
DataDim       = [2000,304]    # Dimension of original one-shot seismic data
data_dsp_blk  = (5,1)         # Downsampling ratio of input
ModelDim      = [201,301]     # Dimension of one velocity model
label_dsp_blk = (1,1)         # Downsampling ratio of output
dh            = 10            # Space interval 

####################################################
####             NETWORK PARAMETERS             ####
####################################################
Epochs        = 6       # Number of epoch
TrainSize = 4 # Number of training set
TestSize      = 1      # Number of testing set
TestBatchSize = 1       #Number of batch testing
start_test = TrainSize+1 #Position of start Test, you can change manually

BatchSize         = 1        # Number of batch size
LearnRate         = 1e-2      # Learning rate
Nclasses          = 1         # Number of output channels
Inchannels        = 29        # Number of input channels, i.e. the number of shots
SaveEpoch         = 2    #Number of epochs where the model will stop.    
DisplayStep       = 2         # Number of steps till outputting stats

