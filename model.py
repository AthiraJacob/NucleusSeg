from keras.layers import Conv2D, Conv2DTranspose, Input, MaxPooling2D, Dropout
from keras.layers import Concatenate, Activation, LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate
from keras.models import Model
from keras import backend as K
import tensorflow as tf

import numpy as np

def side_branch(x, factor):
    x = Conv2D(1, (1, 1), activation=None, padding='same')(x)

    kernel_size = (2*factor, 2*factor)
    x = Conv2DTranspose(1, kernel_size, strides=factor, padding='same', use_bias=False, activation=None)(x)

    return x

def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


def network():
	inputs = Input((256,256, 3))

	fs = 16;

	c1 = Conv2D(fs, (3, 3), padding='same', kernel_initializer = 'glorot_normal', name='block1_conv1')(inputs)
	c1 = LeakyReLU()(c1)
	c1 = Dropout(0.5)(c1)
	c1 = Conv2D(fs, (3, 3),  padding='same', kernel_initializer = 'glorot_normal', name='block1_conv2')(c1)
	c1 = BatchNormalization()(c1)
	c1 = LeakyReLU()(c1)
	p1 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block1_pool')(c1) #128

	c2 = Conv2D(fs*2, (3, 3),  padding='same', kernel_initializer = 'glorot_normal', name='block2_conv1')(p1)
	c2 = LeakyReLU()(c2)
	c2 = Dropout(0.5)(c2)
	c2 = Conv2D(fs*2, (3, 3),  padding='same', kernel_initializer = 'glorot_normal', name='block2_conv2')(c2)
	c2 = BatchNormalization()(c2)
	c2 = LeakyReLU()(c2)
	p2 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block2_pool')(c2) #64

	c3 = Conv2D(fs*4, (3, 3),  padding='same', kernel_initializer = 'glorot_normal', name='block3_conv1')(p2)
	c3 = LeakyReLU()(c3)
	c3 = Dropout(0.5)(c3)
	c3 = Conv2D(fs*4, (3, 3),  padding='same', kernel_initializer = 'glorot_normal', name='block3_conv2')(c3)
	c3 = BatchNormalization()(c3)
	c3 = LeakyReLU()(c3)
	p3 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block3_pool')(c3) #32

	c4 = Conv2D(fs*8, (3, 3),  padding='same', kernel_initializer = 'glorot_normal', name='block4_conv1')(p3)
	c4 = LeakyReLU()(c4)
	c4 = Dropout(0.5)(c4)
	c4 = Conv2D(fs*8, (3, 3),  padding='same', kernel_initializer = 'glorot_normal', name='block4_conv2')(c4)
	c4 = BatchNormalization()(c4)
	c4 = LeakyReLU()(c4)

	u1 = Conv2DTranspose(fs*4, (2, 2), strides=(2, 2), padding='same') (c4)
	u1 = concatenate([u1, c3])
	c5 = Conv2D(fs*4, (3, 3),  kernel_initializer='glorot_normal', padding='same') (u1)
	c5 = LeakyReLU()(c5)
	c5 = Dropout(0.5) (c5)
	c5 = Conv2D(fs*4, (3, 3),  kernel_initializer='glorot_normal', padding='same') (c5) #64x64x64
	b1= side_branch(c5, 4) 
	c5 = BatchNormalization()(c5)
	c5 = LeakyReLU()(c5)

	u2 = Conv2DTranspose(fs*2, (2, 2), strides=(2, 2), padding='same') (c5)
	u2 = concatenate([u2, c2])
	c6 = Conv2D(fs*2, (3, 3),  kernel_initializer='glorot_normal', padding='same') (u2)
	c6 = LeakyReLU()(c6)
	c6 = Dropout(0.5) (c6)
	c6 = Conv2D(fs*2, (3, 3),  kernel_initializer='glorot_normal', padding='same') (c6) #128x128x32
	b2 = side_branch(c6, 2)
	c6 = BatchNormalization()(c6)
	c6 = LeakyReLU()(c6)

	u3 = Conv2DTranspose(fs*4, (2, 2), strides=(2, 2), padding='same') (c6)
	u3 = concatenate([u3, c1])
	c7 = Conv2D(fs, (3, 3),  kernel_initializer='glorot_normal', padding='same') (u3)
	c7 = LeakyReLU()(c7)
	c7 = Dropout(0.5) (c7)
	c7 = Conv2D(fs, (3, 3),  kernel_initializer='glorot_normal', padding='same') (c7) #256x256x16
	b3 = side_branch(c7, 1)
	c7 = BatchNormalization()(c7)
	c7 = LeakyReLU()(c7)


	 # fuse
	fuse = Concatenate(axis=-1)([b1, b2, b3])
	fuse = Conv2D(1, (1,1), padding='same', use_bias=False, activation=None)(fuse) # 256x256x1

    # outputs
	o1    = Activation('sigmoid', name='o1')(b1)
	o2    = Activation('sigmoid', name='o2')(b2)
	o3    = Activation('sigmoid', name='o3')(b3)
	ofuse = Activation('sigmoid', name='ofuse')(fuse)

	model = Model(inputs=[inputs], outputs=[o1, o2, o3,  ofuse])

	model.compile(loss={'o1':'binary_crossentropy','o2':'binary_crossentropy','o3':'binary_crossentropy','ofuse':'binary_crossentropy'}, metrics={'ofuse': mean_iou}, optimizer='adam')

	return model
