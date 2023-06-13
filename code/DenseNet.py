
from keras.layers import Dropout, Activation, Concatenate, Conv2D, AveragePooling2D, BatchNormalization
from keras.regularizers import l2


# 忽略提醒
import warnings

warnings.filterwarnings("ignore")



# 定义密集卷积块中单个卷积层
def conv_factory(x, filters, dropout_rate, weight_decay=1e-4):
    x = Activation('elu')(x)
    x = Conv2D(filters=filters,
               kernel_size=(3, 3),
               kernel_initializer="he_normal",
               padding="same",
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(dropout_rate)(x)
    return x


# 定义transition层
def transition(x, filters, dropout_rate, weight_decay=1e-4):
    # x = Activation('relu')(x)
    x = Conv2D(filters=filters,
               kernel_size=(1, 1),
               kernel_initializer="he_normal",
               padding="same",
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(dropout_rate)(x)
    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = BatchNormalization(axis=-1)(x)
    return x


# 定义密集卷积块
def denseblock(x, layers, filters, growth_rate, dropout_rate=None, weight_decay=1e-4):
    list_feature_map = [x]
    # 循环三次conv_factory部分
    for i in range(layers):
        x = conv_factory(x, growth_rate,
                         dropout_rate, weight_decay)
        list_feature_map.append(x)
        x = Concatenate(axis=-1)(list_feature_map)
        filters = filters + growth_rate
    return x, filters




