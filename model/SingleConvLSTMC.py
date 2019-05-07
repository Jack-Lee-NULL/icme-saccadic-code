
# file: SingleConvLSTMC.py
# author: Jingquan Lee
# date: 2019-04-13
#

import numpy as np
import tensorflow as tf
import cv2

from model.SingleConvLSTMA import SingleConvLSTMA
from model.BasicConvLSTMCell import BasicConvLSTMCell
from model.ModifiedConvLSTMCellB import ModifiedConvLSTMCellB
from ResNet50.resnet import ResNet50

class SingleConvLSTMC(SingleConvLSTMA):

    def _init_cell(self):
        self._cell = ModifiedConvLSTMCellB(shape=self._shape,
            filter_size=self._filter_size,
            num_features=self._c_h_channel[0],
            h_depth=self._c_h_channel[1],
            bias_start=self._forget_bias)

    def _init_holder(self):
        self._c_init = tf.placeholder(tf.float32, 
                (None, self._shape[2], self._shape[3], self._c_h_channel[0]), name='c_init')
        self._h_init = tf.placeholder(tf.float32, 
                (None, self._shape[2], self._shape[3], self._c_h_channel[1]), name='h_init')
        self._o_init = tf.placeholder(tf.float32,
                (None, self._num_steps, self._shape[2], self._shape[3], 1))
        self._inputs_1 = tf.placeholder(tf.float32,
                (None, self._shape[2], self._shape[3], self._inputs_channel[0]))
        self._inputs_2 = tf.placeholder(tf.float32,
                (None, self._shape[2], self._shape[3], self._inputs_channel[0]))
        self._inputs_3 = tf.placeholder(tf.float32,
                (None, self._shape[0], self._shape[1], 3))
        self._inputs = (self._inputs_1, self._inputs_2, self._inputs_3)

    @property
    def o_init(self):
        return self._o_init

    def _attention_net(self, inputs, weight, stage='1'):
        with tf.variable_scope('attention_net_'+stage, reuse=tf.AUTO_REUSE):
            filters1 = tf.get_variable('filter1',
                    shape=[3, 3, inputs.shape.as_list()[3], int(inputs.shape.as_list()[3]//2)],
                    dtype=tf.float32)
            bias1 = tf.get_variable('bias1', 
                    shape=[int(inputs.shape.as_list()[3]//2)], dtype=tf.float32)
            filters2 = tf.get_variable('filter2',
                    shape=[3, 3, 1, int(inputs.shape.as_list()[3]//2)],
                    dtype=tf.float32)
            bias2 = tf.get_variable('bias2', 
                    shape=[int(inputs.shape.as_list()[3]//2)], dtype=tf.float32)
            filters3 = tf.get_variable('filter3',
                    shape=[3, 3, inputs.shape.as_list()[3], int(inputs.shape.as_list()[3]//2)],
                    dtype=tf.float32)
            preds = tf.nn.conv2d(inputs, filters1, strides=[1, 1, 1, 1], padding='SAME')
            preds = tf.sigmoid(preds + bias1)
            weight = tf.nn.conv2d(weight, filters2, strides=[1, 1, 1, 1], padding='SAME')
            weight = tf.sigmoid(weight + bias2)
            inputs_data = tf.nn.conv2d(inputs, filters3, strides=[1, 1, 1, 1], padding='SAME')
            inputs_data = tf.layers.batch_normalization(inputs_data, axis=3, name='inputs_batch')
            preds = preds * weight * inputs_data
            preds = tf.layers.batch_normalization(preds, axis=3, name='batch_norm')
            preds = tf.nn.relu(preds)
            return preds

    def _conv_block(self, input_tensor, filters, stage):
        channels1, channels2, channels3 = filters
        with tf.variable_scope('conv_block_'+stage, reuse=tf.AUTO_REUSE):
            filters1 = tf.get_variable('filter1_'+stage,
                    shape=[5, 5, input_tensor.shape.as_list()[3], channels1],
                    dtype=tf.float32)
            bias1 = tf.get_variable('bias1_'+stage, shape=[channels1],
                    dtype=tf.float32)
            filters2 = tf.get_variable('filter2_'+stage,
                    shape=[5, 5, channels1, channels2],
                    dtype=tf.float32)
            bias2 = tf.get_variable('bias2_'+stage, shape=[channels2],
                    dtype=tf.float32)
            filters3 = tf.get_variable('filter3_'+stage,
                    shape=[5, 5, channels3, channels2],
                    dtype=tf.float32)
            bias3 = tf.get_variable('bias3_'+stage, shape=[channels3],
                    dtype=tf.float32)
            filters4 = tf.get_variable('filter4_'+stage,
                    shape=[5, 5, channels3, input_tensor.shape.as_list()[3]],
                    dtype=tf.float32)
            bias4 = tf.get_variable('bias4_'+stage, shape=[channels3],
                    dtype=tf.float32)

            x = tf.nn.conv2d(input_tensor, filters1,
                    strides=[1, 1, 1, 1], padding='SAME') + bias1
            x = tf.layers.batch_normalization(x, axis=3, name='batch1_'+stage)
            x = tf.nn.relu(x)

            x = tf.nn.conv2d(x, filters2,
                    strides=[1, 1, 1, 1], padding='SAME') + bias2
            x = tf.layers.batch_normalization(x, axis=3, name='batch2_'+stage)
            x = tf.nn.relu(x)

            x = tf.nn.conv2d_transpose(x, filters3, 
                    [self._batch_size, int(x.shape.as_list()[1]*2),int(x.shape.as_list()[2]*2), channels3], 
                    strides=[1, 2, 2, 1], padding='SAME') + bias3
            x = tf.layers.batch_normalization(x, axis=3, name='batch3_'+stage)
            x = tf.nn.relu(x)
        
            shortcut = tf.nn.conv2d_transpose(input_tensor, filters4, 
                    [self._batch_size, int(input_tensor.shape.as_list()[1]*2),
                    int(input_tensor.shape.as_list()[2]*2), channels3], 
                    strides=[1, 2, 2, 1], padding='SAME') + bias4
            shortcut = tf.layers.batch_normalization(shortcut, axis=3, name='batch4_'+stage)
            shortcut = tf.nn.relu(shortcut)
            x = x + shortcut
        return x


    def __call__(self, mode='train'):
        inputs = tf.concat([self._inputs_1, self._inputs_2], axis=3)
        c_all = []
        h_all = []
        preds_all = []
        reconstruct_all = []
        for i in range(self._num_steps):
            if i == 0:
                c = self._c_init
                h = self._h_init
            state = (c, h)
            attention = self._o_init[:, i, :, :, :]
            with tf.variable_scope('IRLSTM_Cell', reuse=tf.AUTO_REUSE):
                merge_feature = self._attention_net(inputs, attention)
                c, h = self._cell(merge_feature, state=state, bias_start=0.0)
                preds = self._conv_block(c, [128, 128, 64], '1')
                preds = self._conv_block(preds, [64, 64, 32], '2')
                filters = tf.get_variable('filters',
                        shape=[7, 7, 3, 32],
                        dtype=tf.float32)
                bias = tf.get_variable('bias', shape=[3], dtype=tf.float32)
                preds = tf.nn.conv2d_transpose(preds, filters, 
                        [self._batch_size, int(preds.shape.as_list()[1]*2),
                        int(preds.shape.as_list()[2]*2), 3],
                        strides=[1, 2, 2, 1], padding='SAME') + bias
                reconstruct = preds
                preds = tf.abs(preds - self._inputs_3)
                #preds = tf.layers.batch_normalization(preds, axis=3, name='batchnormal')
                #preds = tf.nn.relu(preds)
                #preds = tf.concat([preds, self._inputs_3], axis=3)
                filters1 = tf.get_variable('filters1',
                        shape=[3, 3, 3, 1],
                        dtype=tf.float32)
                bias1 = tf.get_variable('bias1', shape=[1], dtype=tf.float32)
                filters2 = tf.get_variable('filters2',
                        shape=[3, 3, 1, 1],
                        dtype=tf.float32)
                bias2 = tf.get_variable('bias2', shape=[1], dtype=tf.float32)
                preds = tf.nn.conv2d(preds, filters1,
                        strides=[1, 1, 1, 1],
                        padding='SAME') + bias1
                preds = tf.layers.batch_normalization(preds, axis=3, name='batchnormal1')
                preds = tf.nn.relu(preds)
                preds = tf.nn.conv2d(preds, filters2,
                        strides=[1, 1, 1, 1],
                        padding='SAME') + bias2
                if mode=='test':
                    preds = tf.sigmoid(preds)
                preds_all.append(tf.expand_dims(preds, axis=1))
                reconstruct_all.append(tf.expand_dims(reconstruct, axis=1))
                c_all = c
                h_all = h
        preds = tf.concat(preds_all, axis=1)
        reconstruct = tf.concat(reconstruct_all, axis=1)
        return  preds, c_all, h_all, reconstruct

