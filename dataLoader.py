import random
import numpy as np
import pdb

class DataSet():
	'''
	Helper functions for data
	'''
	def __init__(self, folder, preprocess):	
		train = np.load(folder + 'train.npy',  encoding = 'latin1').item()
		val = np.load(folder + 'val.npy',  encoding = 'latin1').item()
		test = np.load(folder + 'test.npy',  encoding = 'latin1').item() 
		if preprocess is True:
			train['imgs'] = self.preprocess(train['imgs'])
			val['imgs'] = self.preprocess(val['imgs'])
			test['imgs'] = self.preprocess(test['imgs'])
		self.train = train; self.val = val; self.test = test;
		self.nTrain = train['imgs'].shape[0]
		self.nVal = val['imgs'].shape[0]
		self.nTest = test['imgs'].shape[0]


	def getData(self):
		return self.train, self.val, self.test

	def preprocess(self,imgs ):
		pass
