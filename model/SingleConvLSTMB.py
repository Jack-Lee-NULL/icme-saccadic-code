#
#
# file: SingleConvLSTMB.py
# author: Jingquan Lee
# date: 2019-04-02
#
#

import tensorflow as tf

from model.BasicSaccadicModel import BasicSaccadicModel
from model.BasicConvLSTMCell import BasicConvLSTMCell
from model.ModifiedConvLSTMCellB import ModifiedConvLSTMCellB
from ResNet50.resnet_all import ResNet50

class SingleConvLSTMB(BasicSaccadicModel):

    def _init_cell(self):
        self._cell = ModifiedConvLSTMCellB(shape=self._shape,
                filter_size=self._filter_size,
                num_features=self._c_h_channel[0],
                h_depth=self._c_h_channel[1],
                bias_start=self._forget_bias[0])

    def _init_holder(self):
        self._c_init = tf.placeholder(tf.float32, 
                (self._batch_size, self._shape[2], self._shape[3], self._c_h_channel[0]), name='c_init')
        self._h_init = tf.placeholder(tf.float32, 
                (self._batch_size, self._shape[2], self._shape[3], self._c_h_channel[1]), name='h_init')
        self._o_init = tf.placeholder(tf.float32, 
                (self._batch_size, self._shape[2], self._shape[3], 1), name='o_init')
        self._inputs = tf.placeholder(tf.float32, 
                (self._batch_size, self._shape[0], self._shape[1], self._inputs_channel[0]), name='inputs')
        #self._resnet = ResNet50(include_top=False, input_tensor=self._inputs)

    @property
    def o_init(self):
        return self._o_init

    def __call__(self, mode='train'):
        preds=[]
        c_record = []
        h_record = []
        x_inhibation = self._inputs
        x_inhibation = tf.layers.batch_normalization(x_inhibation, axis=3)
        for i in range(self._num_steps):
            if i == 0:
                c = self._c_init
                h = self._h_init
                o = self._o_init
            x_interest = x_inhibation * o
            scope = 'SCL_B_'
            c, h = self._cell(inputs=(x_interest, x_inhibation), state=(c, h), scope=scope)

            c_record.append(x_interest)
            h_record.append(h)

            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                matrix1 = tf.get_variable(
                        'Out_Matrix1', [self._filter_size[0], self._filter_size[1],
                        self._c_h_channel[1], 1],
                        dtype=tf.float32,
                        initializer=tf.orthogonal_initializer())
                bias1 = tf.get_variable(
                        'Out_Bias1', shape=[1], dtype=tf.float32,
                        initializer=tf.constant_initializer(self._forget_bias[0]))
            o = tf.nn.conv2d(h, matrix1, strides=[1, 1, 1, 1], padding='SAME')
            o = tf.sigmoid(o + bias1)
            pred = tf.expand_dims(o, axis=1)
            preds.append(pred)
        preds = tf.concat(preds, axis=1)
        return preds, c_record, h_record
