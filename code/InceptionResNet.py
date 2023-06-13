
import pandas as pd
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping

from keras.layers import Input, Conv1D, GlobalAvgPool2D, GlobalAveragePooling1D, \
    Dropout, Dense, Activation, Concatenate, Multiply, MaxPool2D, Add, recurrent, \
    LSTM, Bidirectional, Conv2D, AveragePooling2D, BatchNormalization, Flatten, GlobalAveragePooling2D, \
    GlobalMaxPooling2D, Reshape, Permute, multiply, Lambda, add, subtract, MaxPooling2D, GRU, ReLU
from keras.regularizers import l1, l2
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, f1_score
from keras.models import Model, load_model
import tensorflow as tf
from keras import backend as K

from sklearn.model_selection import KFold

from keras.layers import Layer, MaxPooling1D, GaussianNoise

from keras.optimizer_v2.adam import Adam
# 忽略提醒
import warnings

warnings.filterwarnings("ignore")


def convolution_block(x, filters, kernel_size, strides=(1,1), padding='same', activation='relu'):
    x = Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    return x

def inception_resnet_block(x, scale=0.1):
    # 保存输入，用于与残差分支相加
    shortcut = x

    # 第一个分支
    branch_1 = convolution_block(x, 32, (1,1))

    # 第二个分支
    branch_2 = convolution_block(x, 32, (1,1))
    branch_2 = convolution_block(branch_2, 32, (3,3))

    # 第三个分支
    branch_3 = convolution_block(x, 32, (1,1))
    branch_3 = convolution_block(branch_3, 48, (3,3))
    branch_3 = convolution_block(branch_3, 64, (3,3))

    # 将分支连接起来
    x = Concatenate(axis=-1)([branch_1, branch_2, branch_3])

    # 添加残差连接
    x = convolution_block(x, K.int_shape(shortcut)[-1], (1,1), activation=None)
    print(x.shape)
    print(shortcut.shape)
    x = add([x, shortcut])
    x = Activation('relu')(x)
    return x


class GAMAttention(Layer):
    def __init__(self, filters, rate=4, **kwargs):
        self.input_dims = int(filters)
        self.out_dims = int(filters)
        self.reduce_dims = int(filters / rate)
        self.rate = rate

        # channel attention
        self.channel_linear_reduce = Dense(self.reduce_dims,name='channel_Dense_reduce')
        self.channel_activation = Activation('relu', name='channel_activation')
        self.channel_linear = Dense(self.input_dims,name='channel_Dense')

        # spatial attention
        self.spatial_con_reduce = Conv2D(filters=self.reduce_dims, kernel_size=7, padding='same',
                                         name='spatial_con_reduce')
        self.spatial_bn_reduce = BatchNormalization()
        self.spatial_activation = Activation('relu', name='spatial_activation')
        self.spatial_con = Conv2D(filters=self.out_dims, kernel_size=7, padding='same', name='spatial_con')
        self.spatial_bn = BatchNormalization()

        self.in_shape = None
        super(GAMAttention, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_dims": self.input_dims,
            "out_dims": self.out_dims,
            "reduce_dims": self.reduce_dims,
            "rate": self.rate,
            "in_shape": self.in_shape
        })
        return config

    def build(self, input_shape):
        assert input_shape[-1] == self.out_dims, 'input filters must equal to input of channel'
        self.in_shape = input_shape
        # channel attention
        self.channel_linear_reduce.build((input_shape[0], input_shape[1] * input_shape[2], input_shape[3]))
        self._trainable_weights += self.channel_linear_reduce.trainable_weights
        self.channel_linear.build((input_shape[0], input_shape[1] * input_shape[2], self.reduce_dims))
        self._trainable_weights += self.channel_linear.trainable_weights

        # spatial attention
        self.spatial_con_reduce.build(input_shape)
        self._trainable_weights += self.spatial_con_reduce.trainable_weights
        self.spatial_bn_reduce.build((input_shape[0], input_shape[1], input_shape[2], self.reduce_dims))
        self._trainable_weights += self.spatial_bn_reduce.trainable_weights
        self.spatial_con.build((input_shape[0], input_shape[1], input_shape[2], self.reduce_dims))
        self._trainable_weights += self.spatial_con.trainable_weights
        self.spatial_bn.build(input_shape)
        self._trainable_weights += self.spatial_bn.trainable_weights

        super(GAMAttention, self).build(input_shape)  # Be sure to call this at the end

    def call(self, f1, **kwargs):
        # channel attention
        tmp = Reshape((-1, self.input_dims))(f1)
        tmp = self.channel_linear_reduce(tmp)
        tmp = self.channel_activation(tmp)
        tmp = self.channel_linear(tmp)
        mc = Reshape(self.in_shape[1:])(tmp)
        f2 = mc * f1

        # spatial attention
        tmp = self.spatial_con_reduce(f2)
        tmp = self.spatial_bn_reduce(tmp)
        tmp = self.spatial_activation(tmp)
        tmp = self.spatial_con(tmp)
        ms = self.spatial_bn(tmp)
        f3 = ms * f2

        return f3


def build_model(windows=5, weight_decay=1e-4):
    
    input_1 = Input(shape=(windows, 81, 1))

    # Stem block
    x = convolution_block(input_1, 32, (3, 3), strides=(2, 2))
    x = convolution_block(x, 32, (3, 3))
    x = convolution_block(x, 64, (3, 3), padding='same')

    # Inception-ResNet blocks
    for _ in range(5):
        x = inception_resnet_block(x)

    # Top
    x = AveragePooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x)

    print('x',x.shape)
    x_1 = GAMAttention(64)(x)


    # Flatten
    x = Flatten()(x_1)

    # MLP

    x = Dense(units=2, activation="softmax", use_bias=False,
              kernel_initializer='he_normal',
              kernel_regularizer=l2(weight_decay))(x)

    inputs = [input_1]
    outputs = [x]

    model = Model(inputs=inputs, outputs=outputs, name="iPro2L")

    optimizer = Adam(lr=2.2e-4, epsilon=1e-8)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model
