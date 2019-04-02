#
#
# file: SingleConvLSTMA.py
# author: Jingquan Lee
# date: 2019-03-26
#
#

import tensorflow as tf

from model.BasicSaccadicModel import BasicSaccadicModel
from model.ModifiedConvLSTMCellA import ModifiedConvLSTMCellA

class SingleConvLSTMA(BasicSaccadicModel):

    def _init_cell(self):
        self._cell = ModifiedConvLSTMCellA(shape=self._shape,
                filter_size=self._filter_size,
                num_features=self._c_h_channel[0],
                bias_start=self._forget_bias[0], 
                h_depth=self._c_h_channel[1])

    def _init_holder(self):
        self._c_init = tf.placeholder(tf.float32, 
                (None, self._shape[0], self._shape[1], self._c_h_channel[0]), name='c_init')
        self._h_init = tf.placeholder(tf.float32, 
                (None, self._shape[0], self._shape[1], self._c_h_channel[1]), name='h_init')
        self._inputs = tf.placeholder(tf.float32, 
                (None, self._shape[0], self._shape[1], self._inputs_channel[0]), name='inputs')

    def __call__(self):
        preds = []
        for i in range(self._num_steps):
            if i == 0:
               c = self._c_init
               h = self._h_init
            inputs = self._inputs
            scope = 'BSM_'
            c, h = self._cell(inputs, state=(c, h), scope=scope)
            pred = tf.expand_dims(h, axis=1)
            preds.append(pred)
        preds = tf.concat(preds, axis=1)
        return preds
