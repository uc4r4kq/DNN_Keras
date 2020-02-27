# -*- coding: utf-8 -*-
"""
Path setting
@author: Jonas Mendonca (jonas.mendonca@usp.br)
"""

from LibConfig import *
from ParamConfig import *

####################################################
####                   FILENAMES               ####
####################################################

# Data filename
tagD0 = 'georec'
tagV0 = 'vmodel'
tagD1 = 'rec'
tagV1 = 'vmodel'

datafilename  = tagD0
dataname      = tagD1
truthfilename = tagV0
truthname     = tagV1



###################################################
####                   PATHS                  #####
###################################################
main_dir = './'
train_data_dir = '../'
test_data_dir = train_data_dir
folder_dataset = ['georec/','vmodel/']
    
    
## Create Results and Models path
if os.path.exists('./results/') and os.path.exists('./models/'):
    results_dir     = main_dir + 'results/' 
    models_dir      = main_dir + 'models/'
else:
    os.makedirs('./results/')
    os.makedirs('./models/')
    results_dir     = main_dir + 'results/'
    models_dir      = main_dir + 'models/'
    
    
if os.path.exists(results_dir) and os.path.exists(models_dir):  
    results_dir     = results_dir
    models_dir      = models_dir 
else:
    os.makedirs(results_dir)
    os.makedirs(models_dir)
    results_dir     = results_dir
    models_dir      = models_dir

# Create Model name
tagM = 'DNN'
tagM1 = '_TrainSize' + str(TrainSize)
tagM2 = '_Epoch'     + str(Epochs)
tagM3 = '_BatchSize' + str(BatchSize)
tagM4 = '_LR'        + str(LearnRate)

modelname= 'model'
premodelname = 'modelJ'
