#
#
# file: DResConvLSTMC.py
# author: Jingquan Lee
# date: 2019-04-22
#
#

import tensorflow as tf

from model.DResConvLSTM import DResConvLSTM
from model.DResConvLSTMCellC import DResConvLSTMCellC

class DResConvLSTMC(DResConvLSTM):

    def _init_cell(self):
        self._cell = DResConvLSTMCellC(
                filter_size_lr=(self._filter_size[0], self._filter_size[1]),
                filter_size_hr=(self._filter_size[2], self._filter_size[3]),
                shape_lr=(self._shape[0], self._shape[1]),
                shape_hr=(self._shape[2], self._shape[3]),
                c_h_channel=self._c_h_channel,
                forget_bias=self._forget_bias,
                activation=self._activation,
                batch_size=self._batch_size)           

    def _init_holder(self):
        self.lr_c_init = tf.placeholder(tf.float32,
                (None, self._shape[0], self._shape[1],
                self._c_h_channel[0]), name='lr_c_init')
        self.hr_c_init = tf.placeholder(tf.float32,
                (None, self._shape[2], self._shape[3],
                self._c_h_channel[2]), name='hr_c_init')
        self.lr_h_init = tf.placeholder(tf.float32,
                (None, self._shape[0], self._shape[1], self._c_h_channel[1]),
                name='lr_h_init')
        self.hr_h_init = tf.placeholder(tf.float32,
                (None, self._shape[2], self._shape[3], self._c_h_channel[3]), 
                name='hr_h_init')
        self.in_c_init = tf.placeholder(tf.float32,
                (None, 384, 512, self._c_h_channel[3]), 
                name='in_c_init')
        self.in_h_init = tf.placeholder(tf.float32,
                (None, 384, 512, self._c_h_channel[3]), 
                name='in_h_init')
        self.lr_inputs = tf.placeholder(tf.float32,
                (None, self._shape[0], 
                self._shape[1], self._inputs_channel[0]), 
                name='lr_inputs')
        self.hr_inputs = tf.placeholder(tf.float32,
                (None, self._num_steps, self._shape[2], 
                self._shape[3], self._inputs_channel[1]), 
                name='hr_inputs')
        self.in_inputs = tf.placeholder(tf.float32,
                (None, self._num_steps, 384, 
                512, 1), 
                name='in_inputs')

    def __call__(self, mode='train'):
        preds = []
        c_record = []
        h_record = []
        for i in range(self._num_steps):
            if i == 0:
                lr_c = self.lr_c_init
                lr_h = self.lr_h_init
                hr_c = self.hr_c_init
                hr_h = self.hr_h_init
                in_c = self.in_c_init
                in_h = self.in_h_init
                state = (lr_c, lr_h, hr_c, hr_h, in_c, in_h)
                in_inputs = self.in_inputs[:, i, :, :, :]
            inputs = (self.lr_inputs[:, :, :, :],
                    self.hr_inputs[:, i, :, :, :],
                    in_inputs)
            state, pred, in_pred, pred_reconstruct = self._cell(
                    state=state, inputs=inputs, mode=mode)
            in_inputs = 1 - in_pred
            pred = tf.expand_dims(pred, axis=1)
            preds.append(pred)    
            c_record.append(in_pred)
            h_record.append(pred_reconstruct)
        preds = tf.concat(preds, axis=1)
        c_record = tf.concat(c_record, axis=0)
        h_record = tf.concat(h_record, axis=0)
        return preds, c_record, h_record
