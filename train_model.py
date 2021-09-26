# -*- coding: utf-8 -*-
# @Time    : 2021/4/24 18:34
# @File    : train_model.py
from sklearn.model_selection import train_test_split
from keras.models import Model
from tqdm import tqdm # Processing time measurement
from keras import backend as K # The backend give us access to tensorflow operations and allow us to create the Attention class
from keras import optimizers # Allow us to access the Adam class to modify some parameters
from sklearn.model_selection import GridSearchCV, StratifiedKFold # Used to use Kfold to train our model
from keras.callbacks import * # This object helps the model to train in a smarter way, avoiding overfitting
from keras.layers import * # Keras is the most friendly Neural Network library, this Kernel use a lot of layers classes
from keras.models import Model
from sklearn.preprocessing import MinMaxScaler
import inspect
from typing import List
import math

from keras.layers import *
from keras import initializers, regularizers, constraints
from keras.initializers import *
import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import add,Input,Conv1D,Activation,Flatten,Dense


def get_model(model_name, input_shape):
    if model_name == "cnn_lstm":
        model = cnn_lstm(input_shape)
    elif model_name == "capsule_attention":
        model = capsule_attention(input_shape)
    elif model_name == 'lstm_attention':
        model = lstm_attention(input_shape)
    elif model_name == 'tcn':
        model = TCN(input_shape)
    elif model_name == "no_capsule":
        model = no_capsule_attention(input_shape)
    elif model_name == 'vgg':
        model = cnn_vgg(input_shape)
    elif model_name == "lstm1":
        model = lstm1(input_shape)
    elif model_name == "lstm":
        model = lstm1v0(input_shape)
    elif model_name == "lstm2":
        model = lstm2(input_shape)
    elif model_name == "blstm1":
        model = blstm1(input_shape)
    elif model_name == "blstm2":
        model = blstm2(input_shape)
    elif model_name == "lstmfcn":
        model = lstm_fcn(input_shape)
    elif model_name == "resnet":
        model = cnn_resnet(input_shape)
    elif model_name == "mlp":
        model = mlp4(input_shape)
    elif model_name == "lenet":
        model = cnn_lenet(input_shape)
    # elif model_name == "no_capsule":
    #     model = no_capsule_attention(input_shape)
    else:
        print("model name missing")
    return model


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('uniform')
        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape = (input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]
        if self.bias:
            self.b = self.add_weight(shape = (input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None
        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b
        eij = K.tanh(eij)
        a = K.exp(eij)
        if mask is not None:
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim

def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)
    return scale * x

#define our own softmax function instead of K.softmax
def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex/K.sum(ex, axis=axis, keepdims=True)

#A Capsule Implement with Pure Keras
class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, share_weights=True, activation='squash', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights
        if activation == 'squash':
            self.activation = squash
        else:
            self.activation = activations.get(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        #final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        b = K.zeros_like(u_hat_vecs[:,:,:,0]) #shape = [None, num_capsule, input_num_capsule]
        for i in range(self.routings):
            c = softmax(b, 1)
            # o = K.batch_dot(c, u_hat_vecs, [2, 2])
            o = tf.einsum('bin,binj->bij', c, u_hat_vecs)
            if K.backend() == 'theano':
                o = K.sum(o, axis=1)
            if i < self.routings - 1:
                o = K.l2_normalize(o, -1)
                # b = K.batch_dot(o, u_hat_vecs, [2, 3])
                b = tf.einsum('bij,binj->bin', o, u_hat_vecs)
                if K.backend() == 'theano':
                    b = K.sum(b, axis=1)

        return self.activation(o)

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)

def capsule_attention(input_shape):
    inp = Input(shape=(input_shape[1], input_shape[2],))
    x = Bidirectional(GRU(128, return_sequences=True,
                                kernel_initializer=glorot_normal(seed=1029),
                                recurrent_initializer=orthogonal(gain=1.0, seed=1029)))(inp)
    x = Bidirectional(GRU(128, return_sequences=True,
                                kernel_initializer=glorot_normal(seed=1029),
                                recurrent_initializer=orthogonal(gain=1.0, seed=1029)))(x)
    x_1 = Attention(input_shape[1])(x)
    x_1 = Dropout(0.3)(x_1)

    x_2 = Capsule(num_capsule=8, dim_capsule=8, routings=3, share_weights=True)(x)
    x_2 = Flatten()(x_2)
    x_2 = Dropout(0.3)(x_2)

    #     x_rcnn = Conv1D(filters=128,
    #                     kernel_size=1,
    #                     kernel_initializer='he_uniform')(inp)
    n_feature_maps = 128
    conv_x = Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(inp)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)

    conv_y = Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = Activation('relu')(conv_y)

    conv_z = Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(conv_y)
    conv_z = BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(inp)
    shortcut_y = BatchNormalization()(shortcut_y)

    output_block_1 = keras.layers.add([shortcut_y, conv_z])
    output_block_1 = keras.layers.Activation('relu')(output_block_1)

    # BLOCK 2

    conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
    shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

    output_block_2 = keras.layers.add([shortcut_y, conv_z])
    output_block_2 = keras.layers.Activation('relu')(output_block_2)

    # BLOCK 3

    conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_2)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    # no need to expand channels because they are equal
    shortcut_y = keras.layers.BatchNormalization()(output_block_2)

    output_block_3 = keras.layers.add([shortcut_y, conv_z])
    #     output_block_3 = keras.layers.Activation('relu')(output_block_3)

    x_rcnn = Activation('relu')(output_block_3)
    #     x_rcnn = Activation('relu')(x_rcnn)
    x_rcnn_atten = Attention(input_shape[1])(x_rcnn)
    x_rcnn_capsule = Capsule(num_capsule=8, dim_capsule=8, routings=3, share_weights=True)(x_rcnn)
    x_rcnn_capsule = Flatten()(x_rcnn_capsule)

    conc = concatenate([x_1, x_2, x_rcnn_atten, x_rcnn_capsule])
    conc = Dense(512, activation="relu")(conc)
    conc = Dropout(0.3)(conc)
    outp = Dense(1, activation="sigmoid")(conc)
    model = Model(inputs=inp, outputs=outp)
    model.summary()

    return model

#消融
def no_capsule_attention(input_shape):
    inp = Input(shape=(input_shape[1], input_shape[2],))
    x = Bidirectional(LSTM(128, return_sequences=True,
                                kernel_initializer=glorot_normal(seed=1029),
                                recurrent_initializer=orthogonal(gain=1.0, seed=1029)))(inp)
    x = Bidirectional(LSTM(128, return_sequences=True,
                                kernel_initializer=glorot_normal(seed=1029),
                                recurrent_initializer=orthogonal(gain=1.0, seed=1029)))(x)
    x_1 = Attention(input_shape[1])(x)
    x_1 = Dropout(0.3)(x_1)

    # x_2 = Capsule(num_capsule=8, dim_capsule=8, routings=3, share_weights=True)(x)
    # x_2 = Flatten()(x_2)
    # x_2 = Dropout(0.3)(x_2)

    #     x_rcnn = Conv1D(filters=128,
    #                     kernel_size=1,
    #                     kernel_initializer='he_uniform')(inp)
    n_feature_maps = 128
    conv_x = Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(inp)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)

    conv_y = Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = Activation('relu')(conv_y)

    conv_z = Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(conv_y)
    conv_z = BatchNormalization()(conv_z)

    shortcut_y = Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(inp)
    shortcut_y = BatchNormalization()(shortcut_y)

    output_block_1 = keras.layers.add([shortcut_y, conv_z])
    output_block_1 = keras.layers.Activation('relu')(output_block_1)


    conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
    shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

    output_block_2 = keras.layers.add([shortcut_y, conv_z])
    output_block_2 = keras.layers.Activation('relu')(output_block_2)

    # BLOCK 3

    conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_2)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    # no need to expand channels because they are equal
    shortcut_y = keras.layers.BatchNormalization()(output_block_2)

    output_block_3 = keras.layers.add([shortcut_y, conv_z])
    #     output_block_3 = keras.layers.Activation('relu')(output_block_3)

    x_rcnn = Activation('relu')(output_block_3)
    #     x_rcnn = Activation('relu')(x_rcnn)
    x_rcnn_atten = Attention(input_shape[1])(x_rcnn)
    # x_rcnn_capsule = Capsule(num_capsule=8, dim_capsule=8, routings=3, share_weights=True)(x_rcnn)
    # x_rcnn_capsule = Flatten()(x_rcnn_capsule)

    conc = concatenate([x_1, x_rcnn_atten])
    conc = Dense(512, activation="relu")(conc)
    conc = Dropout(0.3)(conc)
    outp = Dense(1, activation="sigmoid")(conc)
    model = Model(inputs=inp, outputs=outp)

    return model

def lstm_attention(input_shape):
    inp = Input(shape=(input_shape[1], input_shape[2],))
    x = Bidirectional(GRU(128, return_sequences=True, kernel_initializer='he_uniform'))(inp)
    x = Bidirectional(GRU(64, return_sequences=True, kernel_initializer='he_uniform'))(x)
    x = Attention(input_shape[1])(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)

    return model

def cnn_lstm(input_shape):
    input_layer = Input(shape=(input_shape[1], input_shape[2],))

    x = Conv1D(32, 8, padding='same', activation='relu', kernel_initializer='he_uniform')(input_layer)
    x = MaxPooling1D(2, padding='same')(x)
    x = Conv1D(64, 8, padding='same', activation='relu', kernel_initializer='he_uniform')(x)
    x = MaxPooling1D(2, padding='same')(x)
    #     x = Conv1D(128, 8, padding='same', activation='relu')(x)
    #     x = MaxPooling1D(2, padding='same')(x)
    #     x = Conv1D(256, 8, padding='same', activation='relu')(x)

    # x = LSTM(64, dropout=0.2, recurrent_dropout=0.5)(x)
    x = LSTM(64, kernel_initializer='he_uniform')(x)
    # x = Flatten(x)
    x = Dense(32, activation="relu")(x)

    output_layer = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

def ResBlock(x, filters, kernel_size, dilation_rate):
    r = Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation_rate, activation='relu')(x)  # 第一卷积
    r = Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation_rate)(r)  # 第二卷积
    if x.shape[-1] == filters:
        shortcut = x
    else:
        shortcut = Conv1D(filters, kernel_size, padding='causal')(x)  # shortcut（捷径）
    o = add([r, shortcut])
    o = Activation('relu')(o)  # 激活函数
    return o

# 序列模型
def TCN(input_shape):
    inputs = Input(shape=(input_shape[1], input_shape[2],))
    x = ResBlock(inputs, filters=64, kernel_size=5, dilation_rate=1)
    x = ResBlock(x, filters=32, kernel_size=5, dilation_rate=2)
    x = ResBlock(x, filters=16, kernel_size=5, dilation_rate=4)
    x = Flatten()(x)
    x = Dense(32, activation="relu")(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(input=inputs, output=x)

    return model


# def mlp4(input_shape, 2):
#     # Z. Wang, W. Yan, T. Oates, "Time Series Classification from Scratch with Deep Neural Networks: A Strong Baseline," Int. Joint Conf. Neural Networks, 2017, pp. 1578-1585
#
#     ip = Input(shape=input_shape)
#     fc = Flatten()(ip)
#
#     fc = Dropout(0.1)(fc)
#
#     fc = Dense(500, activation='relu')(fc)
#     fc = Dropout(0.2)(fc)
#
#     fc = Dense(500, activation='relu')(fc)
#     fc = Dropout(0.2)(fc)
#
#     fc = Dense(500, activation='relu')(fc)
#     fc = Dropout(0.3)(fc)
#
#     out = Dense(2, activation='softmax')(fc)
#
#     model = Model([ip], [out])
#     model.summary()
#     return model

def cnn_lenet(input_shape):
    # Y. Lecun, L. Bottou, Y. Bengio, and P. Haffner, “Gradient-based learning applied to document recognition,” Proceedings of the IEEE, vol. 86, no. 11, pp. 2278–2324, 1998.

    ip = Input(shape=(input_shape[1], input_shape[2],))

    conv = ip

    nb_cnn = int(round(math.log(input_shape[1], 2)) - 3)
    print("pooling layers: %d" % nb_cnn)

    for i in range(nb_cnn):
        conv = Conv1D(6 + 10 * i, 3, padding='same', activation="relu", kernel_initializer='he_uniform')(conv)
        conv = MaxPooling1D(pool_size=2)(conv)

    flat = Flatten()(conv)

    fc = Dense(120, activation='relu')(flat)
    fc = Dropout(0.5)(fc)

    fc = Dense(84, activation='relu')(fc)
    fc = Dropout(0.5)(fc)

    out = Dense(1, activation='sigmoid')(fc)

    model = Model([ip], [out])
    model.summary()
    return model


def cnn_vgg(input_shape):
    # K. Simonyan and A. Zisserman, "Very deep convolutional networks for large-scale image recognition," arXiv preprint arXiv:1409.1556, 2014.

    ip = Input(shape=(input_shape[1], input_shape[2],))

    conv = ip

    nb_cnn = int(round(math.log(input_shape[1], 2)) - 3)
    print("pooling layers: %d" % nb_cnn)

    for i in range(nb_cnn):
        num_filters = min(64 * 2 ** i, 512)
        conv = Conv1D(num_filters, 3, padding='same', activation="relu", kernel_initializer='he_uniform')(conv)
        conv = Conv1D(num_filters, 3, padding='same', activation="relu", kernel_initializer='he_uniform')(conv)
        if i > 1:
            conv = Conv1D(num_filters, 3, padding='same', activation="relu", kernel_initializer='he_uniform')(conv)
        conv = MaxPooling1D(pool_size=2)(conv)

    flat = Flatten()(conv)

    fc = Dense(4096, activation='relu')(flat)
    fc = Dropout(0.5)(fc)

    fc = Dense(4096, activation='relu')(fc)
    fc = Dropout(0.5)(fc)

    out = Dense(1, activation='sigmoid')(fc)

    model = Model([ip], [out])
    model.summary()
    return model


def lstm1v0(input_shape):
    # Original proposal:
    # S. Hochreiter and J. Schmidhuber, “Long Short-Term Memory,” Neural Computation, vol. 9, no. 8, pp. 1735–1780, Nov. 1997.

    ip = Input(shape=(input_shape[1], input_shape[2],))

    l2 = LSTM(512)(ip)
    out = Dense(1, activation='sigmoid')(l2)

    model = Model([ip], [out])

    model.summary()

    return model


def lstm1(input_shape):
    # Original proposal:
    # S. Hochreiter and J. Schmidhuber, “Long Short-Term Memory,” Neural Computation, vol. 9, no. 8, pp. 1735–1780, Nov. 1997.

    # Hyperparameter choices:
    # N. Reimers and I. Gurevych, "Optimal hyperparameters for deep lstm-networks for sequence labeling tasks," arXiv, preprint arXiv:1707.06799, 2017

    ip = Input(shape=(input_shape[1], input_shape[2],))

    l2 = LSTM(100)(ip)
    out = Dense(1, activation='sigmoid')(l2)

    model = Model([ip], [out])

    model.summary()

    return model


def lstm2(input_shape):
    ip = Input(shape=(input_shape[1], input_shape[2],))

    l1 = LSTM(100, return_sequences=True)(ip)
    l2 = LSTM(100)(l1)
    out = Dense(1, activation='sigmoid')(l2)

    model = Model([ip], [out])

    model.summary()

    return model


def blstm1(input_shape):
    # Original proposal:
    # M. Schuster and K. K. Paliwal, “Bidirectional recurrent neural networks,” IEEE Transactions on Signal Processing, vol. 45, no. 11, pp. 2673–2681, 1997.

    # Hyperparameter choices:
    # N. Reimers and I. Gurevych, "Optimal hyperparameters for deep lstm-networks for sequence labeling tasks," arXiv, preprint arXiv:1707.06799, 2017
    ip = Input(shape=(input_shape[1], input_shape[2],))

    l2 = Bidirectional(LSTM(100))(ip)
    out = Dense(1, activation='sigmoid')(l2)

    model = Model([ip], [out])

    model.summary()

    return model


def blstm2(input_shape):
    ip = Input(shape=(input_shape[1], input_shape[2],))

    l1 = Bidirectional(LSTM(100, return_sequences=True))(ip)
    l2 = Bidirectional(LSTM(100))(l1)
    out = Dense(2, activation='softmax')(l2)

    model = Model([ip], [out])

    model.summary()

    return model


def lstm_fcn(input_shape):
    # F. Karim, S. Majumdar, H. Darabi, and S. Chen, “LSTM Fully Convolutional Networks for Time Series Classification,” IEEE Access, vol. 6, pp. 1662–1669, 2018.

    ip = Input(shape=(input_shape[1], input_shape[2],))

    # lstm part is a 1 time step multivariate as described in Karim et al. Seems strange, but works I guess.
    lstm = Permute((2, 1))(ip)

    lstm = LSTM(128)(lstm)
    lstm = Dropout(0.8)(lstm)

    conv = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(ip)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)

    conv = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(conv)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)

    conv = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(conv)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)

    flat = GlobalAveragePooling1D()(conv)

    flat = concatenate([lstm, flat])

    out = Dense(1, activation='sigmoid')(flat)

    model = Model([ip], [out])

    model.summary()

    return model


def cnn_resnet(input_shape):
    # I. Fawaz, G. Forestier, J. Weber, L. Idoumghar, P-A Muller, "Data augmentation using synthetic data for time series classification with deep residual networks," International Workshop on Advanced Analytics and Learning on Temporal Data ECML/PKDD, 2018

    ip = Input(shape=(input_shape[1], input_shape[2],))
    residual = ip
    conv = ip

    for i, nb_nodes in enumerate([64, 128, 128]):
        conv = Conv1D(nb_nodes, 8, padding='same', kernel_initializer="glorot_uniform")(conv)
        conv = BatchNormalization()(conv)
        conv = Activation('relu')(conv)

        conv = Conv1D(nb_nodes, 5, padding='same', kernel_initializer="glorot_uniform")(conv)
        conv = BatchNormalization()(conv)
        conv = Activation('relu')(conv)

        conv = Conv1D(nb_nodes, 3, padding='same', kernel_initializer="glorot_uniform")(conv)
        conv = BatchNormalization()(conv)
        conv = Activation('relu')(conv)

        if i < 2:
            # expands dimensions according to Fawaz et al.
            residual = Conv1D(nb_nodes, 1, padding='same', kernel_initializer="glorot_uniform")(residual)
        residual = BatchNormalization()(residual)
        conv = add([residual, conv])
        conv = Activation('relu')(conv)

        residual = conv

    flat = GlobalAveragePooling1D()(conv)

    out = Dense(1, activation='sigmoid')(flat)

    model = Model([ip], [out])

    model.summary()

    return model


