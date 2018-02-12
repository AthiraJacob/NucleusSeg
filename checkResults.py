import pdb
import numpy as np
import matplotlib.pyplot as plt


dataFold = '../data/'
output_fold = '../output/'

val = np.load(dataFold + 'val.npy',  encoding = 'latin1').item()
test = np.load(dataFold + 'test.npy',  encoding = 'latin1').item() 

val_test = np.load(output_fold + 'val_test.npy',  encoding = 'latin1').item()
test_test = np.load(output_fold + 'test_test.npy',  encoding = 'latin1').item() 




