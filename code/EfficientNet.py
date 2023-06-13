# 引入库文件
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

from keras.layers import Layer, MaxPooling1D, GaussianNoise, DepthwiseConv2D

from keras.optimizer_v2.adam import Adam
# 忽略提醒
import warnings

warnings.filterwarnings("ignore")


# 读取DNA序列
def read_fasta(fasta_file_name):
    seqs = []
    seqs_num = 0
    file = open(fasta_file_name)

    for line in file.readlines():
        if line.strip() == '':
            continue

        if line.startswith('>'):
            seqs_num = seqs_num + 1
            continue
        else:
            seq = line.strip()

            result1 = 'N' in seq
            result2 = 'n' in seq
            if result1 == False and result2 == False:
                seqs.append(seq)
    return seqs


def to_C2_code(seqs):
    properties_code_dict = {
        'A': [0, 0], 'C': [1, 1], 'G': [1, 0], 'T': [0, 1],
        'a': [0, 0], 'c': [1, 1], 'g': [1, 0], 't': [0, 1]
    }
    properties_code = []
    for seq in seqs:
        properties_matrix = np.zeros([2, len(seq)], dtype=float)
        m = 0
        for seq_base in seq:
            properties_matrix[:, m] = properties_code_dict[seq_base]
            m = m + 1
        properties_code.append(properties_matrix)
    return properties_code


# 理化性质编码
def to_properties_code(seqs):
    properties_code_dict = {
        'A': [1, 1, 1], 'C': [0, 1, 0], 'G': [1, 0, 0], 'T': [0, 0, 1],
        'a': [1, 1, 1], 'c': [0, 1, 0], 'g': [1, 0, 0], 't': [0, 0, 1]
    }
    properties_code = []
    for seq in seqs:
        properties_matrix = np.zeros([3, len(seq)], dtype=float)
        m = 0
        for seq_base in seq:
            properties_matrix[:, m] = properties_code_dict[seq_base]
            m = m + 1
        properties_code.append(properties_matrix)
    return properties_code


# 性能评价指标
def show_performance(y_true, y_pred):
    # 定义tp, fp, tn, fn初始值
    TP, FP, FN, TN = 0, 0, 0, 0

    for i in range(len(y_true)):
        if y_true[i] == 1:
            if y_pred[i] > 0.5:
                TP += 1
            else:
                FN += 1
        if y_true[i] == 0:
            if y_pred[i] > 0.5:
                FP += 1
            else:
                TN += 1

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    # 计算敏感性Sn
    Sn = TP / (TP + FN + 1e-06)
    # 计算特异性Sp
    Sp = TN / (FP + TN + 1e-06)
    # 计算Acc值
    Acc = (TP + TN) / len(y_true)
    # 计算MCC：马修斯相关系数是在混淆矩阵环境中建立二元分类器预测质量的最具信息性的单一分数
    MCC = ((TP * TN) - (FP * FN)) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + 1e-06)
    # 计算F1分数
    F1_score = 2 * (precision * recall) / (precision + recall)

    return Sn, Sp, Acc, MCC, F1_score


def performance_mean(performance):
    print('Sn = %.4f ± %.4f' % (np.mean(performance[:, 0]), np.std(performance[:, 0])))
    print('Sp = %.4f ± %.4f' % (np.mean(performance[:, 1]), np.std(performance[:, 1])))
    print('Acc = %.4f ± %.4f' % (np.mean(performance[:, 2]), np.std(performance[:, 2])))
    print('Mcc = %.4f ± %.4f' % (np.mean(performance[:, 3]), np.std(performance[:, 3])))
    print('Auc = %.4f ± %.4f' % (np.mean(performance[:, 4]), np.std(performance[:, 4])))
    print('F1-score = %.4f ± %.4f' % (np.mean(performance[:, 5]), np.std(performance[:, 5])))


def swish_activation(x):
    return x * K.sigmoid(x)

def mbconv_block(inputs, filters, alpha, expansion, output_dim, strides):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    input_dim = K.int_shape(inputs)[channel_axis]

    # 定义扩张比例
    d = input_dim * expansion

    x = Conv2D(d, 1, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation(swish_activation)(x)

    x = DepthwiseConv2D(3, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(swish_activation)(x)

    x = Conv2D(output_dim, 1, padding='same')(x)
    x = BatchNormalization()(x)

    if input_dim == output_dim and strides == 1:
        x = add([x, inputs])

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


def build_model(windows=5, alpha=1.0, layers=3, filters=96,
                growth_rate=32, dropout_rate=0.5, weight_decay=1e-4):
    input_1 = Input(shape=(windows, 81, 1))

    # 定义网络结构
    filters = round(32 * alpha)
    x = Conv2D(filters, 3, strides=(2, 2), padding='same')(input_1)
    x = BatchNormalization()(x)
    x = Activation(swish_activation)(x)

    x = mbconv_block(x, filters, alpha, 1, 6, 2)

    filters = round(16 * alpha)
    x = mbconv_block(x, filters, alpha, 6, 24, 2)
    x = mbconv_block(x, filters, alpha, 6, 24, 1)

    filters = round(32 * alpha)
    x = mbconv_block(x, filters, alpha, 6, 40, 2)
    x = mbconv_block(x, filters, alpha, 6, 40, 1)

    filters = round(48 * alpha)
    x = mbconv_block(x, filters, alpha, 6, 80, 2)
    x = mbconv_block(x, filters, alpha, 6, 80, 1)
    x = mbconv_block(x, filters, alpha, 6, 80, 1)

    filters = round(96 * alpha)
    x = mbconv_block(x, filters, alpha, 6, 112, 1)
    x = mbconv_block(x, filters, alpha, 6, 112, 1)

    filters = round(192 * alpha)
    x = mbconv_block(x, filters, alpha, 6, 320, 1)

    # 添加顶层分类器
    x = Conv2D(round(1280 * alpha), 1, padding='same')(x)

    print('x', x.shape)
    x = GAMAttention(1280)(x)

    x = BatchNormalization()(x)
    x = Activation(swish_activation)(x)
    x = GlobalAveragePooling2D()(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)

    print('x',x.shape)


    # Flatten
    x = Flatten()(x)

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


if __name__ == '__main__':
    # 设置随机种子，很nice
    # 不清楚有啥用？？？？？？？
    np.random.seed(0)
    tf.random.set_seed(1)  # for reproducibility

    # step1：读取数据
    # 读取训练集
    # Read the training set
    train_pos_seqs = np.array(read_fasta('../data/strong.txt'))
    train_neg_seqs = np.array(read_fasta('../data/weak.txt'))

    train_seqs = np.concatenate((train_pos_seqs, train_neg_seqs), axis=0)
    # to_one_hot to_C2_code
    train_onehot = np.array(to_C2_code(train_seqs)).astype(np.float32)

    train_properties_code = np.array(to_properties_code(train_seqs)).astype(np.float32)

    train = np.concatenate((train_onehot, train_properties_code), axis=1)

    train_label = np.array([1] * 1591 + [0] * 1791).astype(np.float32)
    train_label = to_categorical(train_label, num_classes=2)


    # 超参数设置
    BATCH_SIZE = 30
    EPOCHS = 300

    # Cross-validation
    n = 5
    k_fold = KFold(n_splits=n, shuffle=True, random_state=42)

    ten_all_performance = []

    for k in range(10):
        print('*' * 30 + ' the ' + str(k + 1) + ' cycle ' + '*' * 30)

        # 构建模型

        model = build_model()

        all_performance = []

        for fold_count, (train_index, val_index) in enumerate(k_fold.split(train)):
            print('*' * 30 + ' the ' + str(fold_count + 1) + ' fold ' + '*' * 30)

            trains, val = train[train_index], train[val_index]
            trains_label, val_label = train_label[train_index], train_label[val_index]

            # 构建模型

            # model = build_model()

            model.fit(x=trains, y=trains_label, validation_data=(val, val_label), epochs=EPOCHS,
                      batch_size=BATCH_SIZE, shuffle=True,
                      callbacks=[EarlyStopping(monitor='val_loss', patience=20, mode='auto')],
                      verbose=1)



            # model.save('../models/model_' + str(k+1) + '_' + str(fold_count+1) + '.h5')
            #
            # del model
            #
            # model = load_model('../models/model_' + str(k+1) + '_' + str(fold_count+1) + '.h5')

            val_pred = model.predict(val, verbose=1)

            # del model

            # Sn, Sp, Acc, MCC, AUC
            Sn, Sp, Acc, MCC, f1_score = show_performance(val_label[:, 1], val_pred[:, 1])
            AUC = roc_auc_score(val_label[:, 1], val_pred[:, 1])

            print('Sn = %f, Sp = %f, Acc = %f, MCC = %f, AUC = %f, F1-score = %f' % (Sn, Sp, Acc, MCC, AUC, f1_score))

            performance = [Sn, Sp, Acc, MCC, AUC, f1_score]
            all_performance.append(performance)

        del model
        all_performance = np.array(all_performance)
        all_mean_performance = np.mean(all_performance, axis=0)
        ten_all_performance.append(all_mean_performance)


    print('---------------------------------------------10-cycle-mean-result---------------------------------------')
    print(np.array(ten_all_performance))
    performance_mean = performance_mean(np.array(ten_all_performance))
    pd.DataFrame(np.array(ten_all_performance)).to_csv('../files/10_cycle_result.csv',
                                         index=False)




