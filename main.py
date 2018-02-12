    ## Author: Athira

import os
import sys
import argparse
import pdb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from model import network, mean_iou
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model, Model
from dataLoader import DataSet
from keras.utils.generic_utils import get_custom_objects

from keras.callbacks import EarlyStopping, ModelCheckpoint
from dataGenerator import generate_data_generator

CUDA_VISIBLE_DEVICES="0"

FLAGS = None
parser = argparse.ArgumentParser()
parser.add_argument('--nEpochs',type = int, default = 200, help = 'Number of epochs to train for')
parser.add_argument('--restore',type = bool, default = True, help = 'Restore network')
parser.add_argument('--batchSize',type = bool, default = 32, help = 'batchSize for training')
parser.add_argument('--evaluate',type = bool, default = 0, help = 'Evaluate?')
FLAGS, unparsed = parser.parse_known_args()

dataFold = '../data/'
output_fold = '../output/'

print("Loading data..")
data = DataSet(dataFold, preprocess = 0)
train, val, test = data.getData()
nTrain = data.nTrain; nVal = data.nVal; nTest = data.nTest


model_name = output_fold+'model.h5'
s = val['imgs'].shape[1]
nFeatures = 3

input_shape = (s,s,nFeatures)
print(input_shape)

best_model = ModelCheckpoint(model_name, monitor='val_loss', verbose=1, save_best_only=True)

# pdb.set_trace()

get_custom_objects().update({'mean_iou': mean_iou})


if  not FLAGS.evaluate:
    if os.path.isfile(model_name) and FLAGS.restore is True:
        print('------Restoring model-------')
        model = load_model(model_name,custom_objects={'mean_iou': mean_iou})
        model.summary()
        print(input_shape)
    elif ~os.path.isfile(model_name) or FLAGS.restore is False:
    	model = network()



    data_gen_args = dict(rotation_range=90,
            width_shift_range=0.25,
            height_shift_range=0.25,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.1,
            shear_range=0.2,
            fill_mode = 'constant',
            preprocessing_function = None, 
            cval = 0)
    datagen = ImageDataGenerator(**data_gen_args)



    print('---Augmenting images!---')

    model.fit_generator(generate_data_generator(datagen,train['imgs'], train['masks']), steps_per_epoch=len(train['ids']) / FLAGS.batchSize, epochs=FLAGS.nEpochs,
        validation_data=(val['imgs'], [val['masks'],val['masks'],val['masks'],val['masks']]),callbacks=[best_model])

    pdb.set_trace()

else:
    print('Evaluation mode!')
    print('Loading the best model...')
    model = load_model(model_name)
    print('Best Model loaded!')

    score = model.evaluate(val['imgs'], [val['masks'],val['masks'],val['masks'],val['masks']], verbose=0)
    print('Validation loss:', score)

    pdb.set_trace()

    pred_test = model.predict(test['imgs'])
    np.save(output_fold + 'pred_test.npy',pred_test)

    pred_val = model.predict(val['imgs'])
    np.save(output_fold + 'pred_val.npy',pred_val)

