#
#
# file: SingleConvLSTMB4Test.py
# author: Jingquan Lee
# date: 2019-04-03
#
#

import tensorflow as tf

from model.BasicSaccadicModel import BasicSaccadicModel
from model.ModifiedConvLSTMCellB import ModifiedConvLSTMCellB
from model.SingleConvLSTMB import SingleConvLSTMB
from ResNet50.resnet import ResNet50

class SingleConvLSTMB4Test(SingleConvLSTMB):

    def _init_holder(self):
        self._c_init = tf.placeholder(tf.float32, 
                (None, self._shape[0], self._shape[1], self._c_h_channel[0]), name='c_init')
        self._h_init = tf.placeholder(tf.float32, 
                (None, self._shape[0], self._shape[1], self._c_h_channel[1]), name='h_init')
        self._o_init = tf.placeholder(tf.float32, 
                (None, self._shape[0], self._shape[1], 1), name='h_init')
        self._inputs = tf.placeholder(tf.float32, 
                (None, self._shape[0], self._shape[1], self._inputs_channel[0]), name='inputs')

    def __call__(self):
        c = self._c_init
        h = self._h_init
        o = self._o_init
        x_interest = self._resnet(self._inputs * o)
        x_inhibation = self._resnet(self._inputs)
        scope = 'SCL_B_'
        c, h = self._cell(inputs=x_interest, state=(c, h), scope=scope)

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            matrix1 = tf.get_variable(
                    'Out_Matrix1', [self._filter_size[0], self._filter_size[1],
                    x_inhibation.shape[3], self._c_h_channel[0]],
                    dtype=tf.float32,
                    initializer=tf.orthogonal_initializer())
            bias1 = tf.get_variable(
                    'Out_Bias1', self._c_h_channel[0], dtype=tf.float32,
                    initializer=tf.constant_initializer(self._forget_bias[0]))
            matrix2 = tf.get_variable(
                    'Out_Matrix2', [self._filter_size[0], self._filter_size[1],
                    self._c_h_channel[0]*2, 1],
                    dtype=tf.float32,
                    initializer=tf.orthogonal_initializer())
            bias2 = tf.get_variable(
                    'Out_Bias2', [1], dtype=tf.float32,
                    initializer=tf.constant_initializer(self._forget_bias[0]))
        o = tf.nn.conv2d(x_inhibation, matrix1, strides=[1, 1, 1, 1], padding='SAME')
        o = tf.tanh(o + bias1)
        o = tf.concat([h, o], axis=3)
        o = tf.nn.conv2d(o, matrix2, strides=[1, 1, 1, 1], padding='SAME')
        o = tf.sigmoid(o + bias2)
        return c, h, o


