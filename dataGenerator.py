#Functions for data augmentation and generation
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import pdb
import numpy as np
import matplotlib.pyplot as plt
import cv2
from random import randint
import re
from scipy import linalg
import scipy.ndimage as ndi
from six.moves import range
import os
import threading
import warnings
import multiprocessing.pool
from functools import partial

from keras.utils.data_utils import *
from keras import backend as K




try:
	from PIL import Image as pil_image
except ImportError:
	pil_image = None


if pil_image is not None:
	_PIL_INTERPOLATION_METHODS = {
		'nearest': pil_image.NEAREST,
		'bilinear': pil_image.BILINEAR,
		'bicubic': pil_image.BICUBIC,
	}
	# These methods were only introduced in version 3.4.0 (2016).
	if hasattr(pil_image, 'HAMMING'):
		_PIL_INTERPOLATION_METHODS['hamming'] = pil_image.HAMMING
	if hasattr(pil_image, 'BOX'):
		_PIL_INTERPOLATION_METHODS['box'] = pil_image.BOX
	# This method is new in version 1.1.3 (2013).
	if hasattr(pil_image, 'LANCZOS'):
		_PIL_INTERPOLATION_METHODS['lanczos'] = pil_image.LANCZOS


def random_rotation(x, rg, row_axis=1, col_axis=2, channel_axis=0,
					fill_mode='nearest', cval=0.):
	"""Performs a random rotation of a Numpy image tensor.
	# Arguments
		x: Input tensor. Must be 3D.
		rg: Rotation range, in degrees.
		row_axis: Index of axis for rows in the input tensor.
		col_axis: Index of axis for columns in the input tensor.
		channel_axis: Index of axis for channels in the input tensor.
		fill_mode: Points outside the boundaries of the input
			are filled according to the given mode
			(one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
		cval: Value used for points outside the boundaries
			of the input if `mode='constant'`.
	# Returns
		Rotated Numpy image tensor.
	"""
	theta = np.deg2rad(np.random.uniform(-rg, rg))
	rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
								[np.sin(theta), np.cos(theta), 0],
								[0, 0, 1]])

	h, w = x.shape[row_axis], x.shape[col_axis]
	transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
	x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
	return x


def random_shift(x, wrg, hrg, row_axis=1, col_axis=2, channel_axis=0,
				 fill_mode='nearest', cval=0.):
	"""Performs a random spatial shift of a Numpy image tensor.
	# Arguments
		x: Input tensor. Must be 3D.
		wrg: Width shift range, as a float fraction of the width.
		hrg: Height shift range, as a float fraction of the height.
		row_axis: Index of axis for rows in the input tensor.
		col_axis: Index of axis for columns in the input tensor.
		channel_axis: Index of axis for channels in the input tensor.
		fill_mode: Points outside the boundaries of the input
			are filled according to the given mode
			(one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
		cval: Value used for points outside the boundaries
			of the input if `mode='constant'`.
	# Returns
		Shifted Numpy image tensor.
	"""
	h, w = x.shape[row_axis], x.shape[col_axis]
	tx = np.random.uniform(-hrg, hrg) * h
	ty = np.random.uniform(-wrg, wrg) * w
	translation_matrix = np.array([[1, 0, tx],
								   [0, 1, ty],
								   [0, 0, 1]])

	transform_matrix = translation_matrix  # no need to do offset
	x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
	return x


def random_shear(x, intensity, row_axis=1, col_axis=2, channel_axis=0,
				 fill_mode='nearest', cval=0.):
	"""Performs a random spatial shear of a Numpy image tensor.
	# Arguments
		x: Input tensor. Must be 3D.
		intensity: Transformation intensity in degrees.
		row_axis: Index of axis for rows in the input tensor.
		col_axis: Index of axis for columns in the input tensor.
		channel_axis: Index of axis for channels in the input tensor.
		fill_mode: Points outside the boundaries of the input
			are filled according to the given mode
			(one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
		cval: Value used for points outside the boundaries
			of the input if `mode='constant'`.
	# Returns
		Sheared Numpy image tensor.
	"""
	shear = np.deg2rad(np.random.uniform(-intensity, intensity))
	shear_matrix = np.array([[1, -np.sin(shear), 0],
							 [0, np.cos(shear), 0],
							 [0, 0, 1]])

	h, w = x.shape[row_axis], x.shape[col_axis]
	transform_matrix = transform_matrix_offset_center(shear_matrix, h, w)
	x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
	return x


def random_zoom(x, zoom_range, row_axis=1, col_axis=2, channel_axis=0,
				fill_mode='nearest', cval=0.):
	"""Performs a random spatial zoom of a Numpy image tensor.
	# Arguments
		x: Input tensor. Must be 3D.
		zoom_range: Tuple of floats; zoom range for width and height.
		row_axis: Index of axis for rows in the input tensor.
		col_axis: Index of axis for columns in the input tensor.
		channel_axis: Index of axis for channels in the input tensor.
		fill_mode: Points outside the boundaries of the input
			are filled according to the given mode
			(one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
		cval: Value used for points outside the boundaries
			of the input if `mode='constant'`.
	# Returns
		Zoomed Numpy image tensor.
	# Raises
		ValueError: if `zoom_range` isn't a tuple.
	"""
	if len(zoom_range) != 2:
		raise ValueError('`zoom_range` should be a tuple or list of two floats. '
						 'Received arg: ', zoom_range)

	if zoom_range[0] == 1 and zoom_range[1] == 1:
		zx, zy = 1, 1
	else:
		zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
	zoom_matrix = np.array([[zx, 0, 0],
							[0, zy, 0],
							[0, 0, 1]])

	h, w = x.shape[row_axis], x.shape[col_axis]
	transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w)
	x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
	return x


def random_channel_shift(x, intensity, channel_axis=0):
	x = np.rollaxis(x, channel_axis, 0)
	min_x, max_x = np.min(x), np.max(x)
	channel_images = [np.clip(x_channel + np.random.uniform(-intensity, intensity), min_x, max_x)
					  for x_channel in x]
	x = np.stack(channel_images, axis=0)
	x = np.rollaxis(x, 0, channel_axis + 1)
	return x


def transform_matrix_offset_center(matrix, x, y):
	o_x = float(x) / 2 + 0.5
	o_y = float(y) / 2 + 0.5
	offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
	reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
	transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
	return transform_matrix


def apply_transform(x,
					transform_matrix,
					channel_axis=0,
					fill_mode='nearest',
					cval=0.):
	"""Apply the image transformation specified by a matrix.
	# Arguments
		x: 2D numpy array, single image.
		transform_matrix: Numpy array specifying the geometric transformation.
		channel_axis: Index of axis for channels in the input tensor.
		fill_mode: Points outside the boundaries of the input
			are filled according to the given mode
			(one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
		cval: Value used for points outside the boundaries
			of the input if `mode='constant'`.
	# Returns
		The transformed version of the input.
	"""
	x = np.rollaxis(x, channel_axis, 0)
	final_affine_matrix = transform_matrix[:2, :2]
	final_offset = transform_matrix[:2, 2]
	channel_images = [ndi.interpolation.affine_transform(
		x_channel,
		final_affine_matrix,
		final_offset,
		order=0,
		mode=fill_mode,
		cval=cval) for x_channel in x]
	x = np.stack(channel_images, axis=0)
	x = np.rollaxis(x, 0, channel_axis + 1)
	return x


def flip_axis(x, axis):
	x = np.asarray(x).swapaxes(axis, 0)
	x = x[::-1, ...]
	x = x.swapaxes(0, axis)
	return x



def augment(img,mask):
	"""Randomly augment a single image tensor.
	# Arguments
		x: 3D tensor, single image.
		seed: random seed.
	# Returns
		A randomly transformed version of the input (same shape).
	"""
	# x is a single image, so it doesn't have image number at index 0
	img_row_axis = 0
	img_col_axis = 1
	img_channel_axis = 2


	# use composition of homographies
	# to generate final transform that needs to be applied
	rotation_range = 90
	if rotation_range:
		theta = np.deg2rad(np.random.uniform(-rotation_range, rotation_range))
	else:
		theta = 0

	height_shift_range = 0.2
	if height_shift_range:
		tx = np.random.uniform(-height_shift_range, height_shift_range)
		if height_shift_range < 1:
			tx *= img.shape[img_row_axis]
	else:
		tx = 0

	width_shift_range = 0.2
	if width_shift_range:
		ty = np.random.uniform(-width_shift_range, width_shift_range)
		if width_shift_range < 1:
			ty *= img.shape[img_col_axis]
	else:
		ty = 0

	shear_range = 0.05
	if shear_range:
		shear = np.deg2rad(np.random.uniform(-shear_range, shear_range))
	else:
		shear = 0

	zoom_range = [0.2,0.2]
	if zoom_range[0] == 1 and zoom_range[1] == 1:
		zx, zy = 1, 1
	else:
		zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)

	transform_matrix = None
	if theta != 0:
		rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
									[np.sin(theta), np.cos(theta), 0],
									[0, 0, 1]])
		transform_matrix = rotation_matrix

	if tx != 0 or ty != 0:
		shift_matrix = np.array([[1, 0, tx],
								 [0, 1, ty],
								 [0, 0, 1]])
		transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)

	if shear != 0:
		shear_matrix = np.array([[1, -np.sin(shear), 0],
								[0, np.cos(shear), 0],
								[0, 0, 1]])
		transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)

	if zx != 1 or zy != 1:
		zoom_matrix = np.array([[zx, 0, 0],
								[0, zy, 0],
								[0, 0, 1]])
		transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)

	fill_mode = 'constant'
	if transform_matrix is not None:
		h, w = img.shape[img_row_axis], img.shape[img_col_axis]
		transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
		img = apply_transform(img, transform_matrix, img_channel_axis,
							fill_mode=fill_mode, cval=0)
		mask = apply_transform(mask, transform_matrix, img_channel_axis,
							fill_mode=fill_mode, cval=0)

	horizontal_flip = True
	vertical_flip = True
	if horizontal_flip:
		if np.random.random() < 0.5:
			img = flip_axis(img, img_col_axis)
			mask = flip_axis(mask, img_col_axis)

	if vertical_flip:
		if np.random.random() < 0.5:
			img = flip_axis(img, img_row_axis)
			mask = flip_axis(mask, img_row_axis)

	return img,mask
	


def generator(imgs, masks, batchSize):
	n = imgs.shape[0]
	imgs_batch = np.zeros(shape = (batchSize,) + imgs.shape)
	masks_batch = np.zeros(shape = (batchSize,) + masks.shape)
	while 1:
		for i in range(batchSize):
			t = randint(0,n-1)
			img = imgs[t,:,:,:]
			mask = masks[t,:,:,:]
			imgs_batch[i,:,:,:],masks_batch[i,:,:,:] = augment(img, mask)
		yield (imgs_batch,[masks_batch,masks_batch,masks_batch,masks_batch])


def generate_data_generator(datagen, X, Y):
    genX = datagen.flow(X, seed=7)
    genY = datagen.flow(Y, seed=7)
    while True:
            Xi = genX.next()
            Yi1 = genY.next()
            yield Xi, [Yi1, Yi1, Yi1, Yi1]