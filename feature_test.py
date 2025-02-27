import scipy.io
import numpy as np
import os
import pickle
import h5py

# 文件路径
mat_files_path_1 = 'D:/python object/federate learning/FedVC_eeg/FEATUREtest.h5'
mat_files_path_2 = 'D:/python object/federate learning/FedVC_eeg/data/SEED/SEED_ORIGINAL/ExtractedFeatures/2_20140404.mat'



data1 = scipy.io.loadmat(mat_files_path_1)
data2 = scipy.io.loadmat(mat_files_path_2)

a = 1