# Script to read data and write as matrix for easy reading

import os
import sys
import argparse
import pdb
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
import cv2
from utils import create_mask, resize_img, normalize_img

# import pandas as pd

train_fold = '../data/train_imgs' 
test_fold = '../data/test_1_imgs'
out_fold = '../data'

train_ids = next(os.walk(train_fold))[1]
test_ids = next(os.walk(test_fold))[1]

total_train = len(train_ids)
n_train = 600
n_test = len(test_ids)

idx = range(total_train)
shuffle(idx)

train_imgs = np.zeros(shape = (n_train, 256, 256, 3))
train_masks = np.zeros(shape = (n_train, 256, 256,1))
val_imgs = np.zeros(shape = (total_train - n_train, 256, 256, 3))
val_masks = np.zeros(shape = (total_train - n_train, 256, 256, 1))
test_imgs = np.zeros(shape = (n_test, 256, 256, 3))
tr_ids = []
val_ids = []
te_ids = []
orig_sh_train = []
orig_sh_val = []
orig_sh_test = []

k = 0
IMG_HEIGHT = 256
IMG_WIDTH = 256
list_shapes = []

print('Reading train and val..')
for i in range(total_train):
	print(i)
	img_path = train_fold + '/' + train_ids[i] + '/images/' + train_ids[i] + '.png'
	img = cv2.imread(img_path,1)  #256 x 256 x 3?
	# pdb.set_trace()
	mask = create_mask(train_fold + '/' + train_ids[i], shape = (img.shape[0], img.shape[1],1))
	print(mask.shape)
	old_shape = img.shape
	img = np.dstack([normalize_img(img[:,:,q]) for q in range(3) ])
	if (img.shape != (IMG_HEIGHT, IMG_WIDTH, 3)): # resize it
		img = resize_img(img, shape = (IMG_HEIGHT, IMG_WIDTH, 3))
		mask = resize_img(mask, shape = (IMG_HEIGHT, IMG_WIDTH,1))
		# if img.shape in list_shapes:
		# 	pass
		# else:
		# 	list_shapes.append(img.shape)
	# pdb.set_trace()
	# cv2.imshow('image',img)
	# cv2.waitKey(0)
	# cv2.imshow('mask',mask)
	# cv2.waitKey(0)
	# pdb.set_trace()

	if i<n_train:
		train_imgs[i,:,:,:] = img
		train_masks[i,:,:,:] = mask
		tr_ids.append(train_ids[i])
		orig_sh_train.append(old_shape)
	else:
		val_imgs[k,:,:,:] = img
		val_masks[k,:,:,:] = mask
		val_ids.append(train_ids[i])
		orig_sh_val.append(old_shape)
		k = k+1

pdb.set_trace()
print('Saving train and val..')
train = {'imgs':train_imgs,'masks':train_masks, 'ids':tr_ids, 'orig_shapes':orig_sh_train}
val = {'imgs':val_imgs,'masks':val_masks, 'ids':val_ids, 'orig_shapes':orig_sh_val}
np.save(out_fold + '/train.npy',train)
np.save(out_fold + '/val.npy',val)


print('Reading test..')

for i in range(n_test):
	img_path = test_fold + '/' + test_ids[i] + '/images/' + test_ids[i] + '.png'
	img = cv2.imread(img_path,1)  #256 x 256 x 3?
	old_shape = img.shape
	img = np.dstack([normalize_img(img[:,:,q]) for q in range(3) ])
	if (img.shape != (IMG_HEIGHT, IMG_WIDTH, 3)): # resize it
		img = resize_img(img, shape = (IMG_HEIGHT, IMG_WIDTH, 3))
	# cv2.imshow('image',img)
	# cv2.waitKey(0)
	test_imgs[i,:,:,:] = img
	te_ids.append(test_ids[i])
	orig_sh_test.append(old_shape)

pdb.set_trace()
test = {'imgs':test_imgs, 'ids':te_ids, 'orig_shapes':orig_sh_test}

print('Saving test..')
np.save(out_fold + '/test.npy',test)

