import os
import sys
import pdb
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
from skimage.morphology import label

# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)

def normalize_img(img):
	img = np.float32(img)
	img = img - np.min(img)
	img = img/np.max(img)

	return img

def create_mask(folder, shape):
	fold = folder + '/masks/'
	ids = os.listdir(fold)
	mask = np.zeros(shape = (shape[0], shape[1]))
	# pdb.set_trace()
	for m in ids:
		mask_ = cv2.imread(fold+'/'+m,0)
		mask_ = normalize_img(mask_)
		if np.min(mask_)!=0 or np.max(mask_) != 1:
			print('Error in mask range!')
		mask = np.maximum(mask, mask_)

	return mask.reshape(shape)

def resize_img(img, shape):
	old_shape = img.shape
	if old_shape[0]!=old_shape[1]:
		if old_shape[0]>old_shape[1]:
			l_dim = old_shape[0]
			img2 = np.zeros(shape = (l_dim, l_dim,old_shape[2]))
			img2[:,0:old_shape[1],:] = img
		else:
			l_dim = old_shape[1]
			img2 = np.zeros(shape = (l_dim, l_dim, old_shape[2]))
			img2[0:old_shape[0],:,:] = img
	else:
		img2 = img
	# pdb.set_trace()
	if shape[2]==1:
		img3 = np.dstack([cv2.resize(img2[:,:,i],(shape[0],shape[1]), interpolation = cv2.INTER_AREA) for i in range(img2.shape[2]) ])
	else:
		img3 = np.dstack([cv2.resize(img2[:,:,i],(shape[0],shape[1]), interpolation = cv2.INTER_CUBIC) for i in range(img2.shape[2]) ])
	return img3



def resize_img_upample(img,old_shape):
	shape = img.shape
	img3 = np.zeros(shape = (old_shape))
	if old_shape[0]!=old_shape[1]:
		if old_shape[0]>old_shape[1]:
			l_dim = old_shape[0]
			if shape[2]==1:
				img2 = np.dstack([cv2.resize(img[:,:,i],(l_dim,l_dim), interpolation = cv2.INTER_AREA) for i in range(img.shape[2]) ])
			else:
				img2 = np.dstack([cv2.resize(img[:,:,i],(l_dim,l_dim), interpolation = cv2.INTER_CUBIC) for i in range(img.shape[2]) ]) 
			img3 = img2[:,0:old_shape[1],:]
		else:
			l_dim = old_shape[1]
			if shape[2]==1:
				img2 = np.dstack([cv2.resize(img[:,:,i],(l_dim,l_dim), interpolation = cv2.INTER_AREA) for i in range(img.shape[2]) ])
			else:
				img2 = np.dstack([cv2.resize(img[:,:,i],(l_dim,l_dim), interpolation = cv2.INTER_CUBIC) for i in range(img.shape[2]) ]) 
			img3 = img2[0:old_shape[0],:,:]
	else:
		if shape[2]==1:
			img3 = np.dstack([cv2.resize(img[:,:,i],(old_shape[0],old_shape[1]), interpolation = cv2.INTER_AREA) for i in range(img.shape[2]) ])
		else:
			img3 = np.dstack([cv2.resize(img[:,:,i],(old_shape[0],old_shape[1]), interpolation = cv2.INTER_CUBIC) for i in range(img.shape[2]) ]) 
	return img3

