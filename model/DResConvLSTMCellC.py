#
#
# file: DResConvLSTMCellC.py
# author: Jingquan Lee
# date: 2019-04-22
#
#

import tensorflow as tf

from model.BasicConvLSTMCell import BasicConvLSTMCell
from model.DResConvLSTMCell import DResConvLSTMCell
from model.ModifiedConvLSTMCellB import ModifiedConvLSTMCellB

class DResConvLSTMCellC(DResConvLSTMCell):

    def _init_cell(self):
        self._lr_cell = ModifiedConvLSTMCellB(self._shape_lr, self._filter_size_lr,
                num_features=self._c_h_channel[0], h_depth=self._c_h_channel[1],
                bias_start=self._forget_bias[0])
        self._hr_cell = ModifiedConvLSTMCellB(self._shape_hr, self._filter_size_hr,
                num_features=self._c_h_channel[2], h_depth=self._c_h_channel[3],
                bias_start=self._forget_bias[1])
        self._inhibation_cell = ModifiedConvLSTMCellB([384, 512], self._filter_size_hr,
                num_features=self._c_h_channel[2], h_depth=self._c_h_channel[3],
                bias_start=self._forget_bias[1])

    def __call__(self, state, inputs, scope='DR_CONV_LSTM', keep_prob=0.8, mode='train'):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            if mode == 'train':
                lr_inputs = tf.nn.dropout(inputs[0], keep_prob)
                hr_inputs = tf.nn.dropout(inputs[1], keep_prob)
            else:
                lr_inputs = tf.layers.batch_normalization(inputs[0], axis=3, name='inputs_normal')
                hr_inputs = tf.layers.batch_normalization(inputs[1], axis=3, name='inputs_normal')
            lr_state, lr_preds = self._run_lr_cell(
                state=(state[0], state[1]), inputs=lr_inputs,
                keep_prob=keep_prob, scope=scope+'_lr', mode=mode)
            hr_state, hr_preds = self._run_hr_cell(
                state=(state[2], state[3]), inputs=hr_inputs,
                keep_prob=keep_prob, scope=scope+'_hr', mode=mode)
            in_state, in_preds = self._run_inhibation_cell(
                state=(state[4], state[5]), inputs=inputs[2],
                keep_prob=keep_prob, scope=scope+'_in', mode=mode)
            anchor = hr_preds

            #lr_preds = tf.layers.batch_normalization(lr_preds, axis=3, name='preds_normal')
            #hr_preds = tf.layers.batch_normalization(hr_preds, axis=3, name='preds_normal')
            hr_preds = tf.layers.flatten(hr_preds)
            lr_preds = tf.layers.flatten(lr_preds)
            preds = tf.concat([lr_preds, hr_preds], axis=1)

            matrix1 = tf.get_variable('out_matrix1',
                    shape=[preds.shape.as_list()[1], self._shape_lr[0]*self._shape_lr[1]*self._c_h_channel[0]/4],
                    dtype=tf.float32)
            bias1 = tf.get_variable('out_bias1',
                    shape=[self._shape_lr[0]*self._shape_lr[1]*self._c_h_channel[0]/4],
                    dtype=tf.float32,
                    initializer=tf.constant_initializer(0.0))
            matrix4 = tf.get_variable('out_matrix4',
                    shape=[self._shape_lr[0]*self._shape_lr[1]*self._c_h_channel[0]/4, 
                    self._shape_lr[0]*self._shape_lr[1]*self._c_h_channel[0]],
                    dtype=tf.float32)
            bias4 = tf.get_variable('out_bias4',
                    shape=[self._shape_lr[0]*self._shape_lr[1]*self._c_h_channel[0]],
                    dtype=tf.float32,
                    initializer=tf.constant_initializer(0.0))

            preds = tf.matmul(preds, matrix1)
            preds = tf.nn.relu(preds + bias1)
            preds = tf.matmul(preds, matrix4)
            preds = tf.nn.relu(preds + bias4)
            preds = tf.reshape(preds, shape=[-1, int(self._shape_lr[0]), 
                    int(self._shape_lr[1]), self._c_h_channel[0]])

            matrix2 = tf.get_variable('out_matrix2',
                    shape=[7, 7, int(self._c_h_channel[0]//2), self._c_h_channel[0]],
                    dtype=tf.float32,
                    initializer=tf.random_normal_initializer())
            bias2 = tf.get_variable('out_bias2',
                    shape=[int(self._c_h_channel[0]//2)],
                    dtype=tf.float32,
                    initializer=tf.constant_initializer(0.0))

            matrix3 = tf.get_variable('out_matrix3',
                    shape=[5, 5, 1, int(self._c_h_channel[0]//2)],
                    dtype=tf.float32)
            bias3 = tf.get_variable('out_bias3',
                    shape=[1],
                    dtype=tf.float32,
                    initializer=tf.constant_initializer(0.0))

            matrix5 = tf.get_variable('out_matrix5',
                    shape=[5, 5, int(self._c_h_channel[0]), 1],
                    dtype=tf.float32)
            bias5 = tf.get_variable('out_bias5',
                    shape=[1],
                    dtype=tf.float32,
                    initializer=tf.constant_initializer(0.0))

            matrix6 = tf.get_variable('out_matrix6',
                    shape=[5, 5, 1, 1],
                    dtype=tf.float32)
            bias6 = tf.get_variable('out_bias6',
                    shape=[1],
                    dtype=tf.float32,
                    initializer=tf.constant_initializer(0.0))
            matrix7 = tf.get_variable('out_matrix7',
                    shape=[5, 5, 1, 1],
                    dtype=tf.float32)
            bias7 = tf.get_variable('out_bias7',
                    shape=[1],
                    dtype=tf.float32,
                    initializer=tf.constant_initializer(0.0))
            

            preds = tf.nn.conv2d_transpose(preds, matrix2, 
                    [self._batch_size, int(self._shape_lr[0]*4), 
                    int(self._shape_lr[1]*4), int(self._c_h_channel[0]//2)],
                    strides=[1, 4, 4, 1])
            preds = tf.nn.relu(preds+bias2)
            preds = tf.layers.batch_normalization(preds, axis=3, name='out_preds')

            preds = tf.nn.conv2d_transpose(preds, matrix3, 
                    [self._batch_size, int(self._shape_lr[0]*8), 
                    int(self._shape_lr[1]*8), 1],
                    strides=[1, 2, 2, 1])
            preds = tf.nn.tanh(preds+bias3)

            in_preds = tf.nn.conv2d(in_preds, matrix5, strides=[1, 1, 1, 1], padding='SAME')
            in_preds = tf.sigmoid(in_preds + bias5)
            in_preds = 1 - in_preds
            in_preds = tf.multiply(preds, in_preds)

            a_preds = tf.nn.conv2d(preds, matrix6, strides=[1, 1, 1, 1], padding='SAME')
            a_preds = tf.sigmoid(a_preds + bias6)
            a_preds = tf.multiply(a_preds, in_preds)
            
            a_preds = tf.nn.conv2d(a_preds, matrix7, strides=[1, 1, 1, 1], padding='SAME')
            a_preds = tf.tanh(a_preds + bias7)
            preds = a_preds + preds
            
            

            state = (lr_state[0], lr_state[1], hr_state[0], hr_state[1], in_state[0], in_state[1])
            if mode == 'test':
                preds = tf.sigmoid(preds)

            return state, preds, in_preds, anchor

    def _run_lr_cell(self, state, inputs, keep_prob, scope='DR_CONV_LSTM', mode='train'):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            if mode == 'train':
                state = (tf.nn.dropout(state[0], keep_prob), tf.nn.dropout(state[1], keep_prob))
            lr_c, lr_h = self._lr_cell(inputs, state=state, scope=scope+'_lr')
            new_state = (lr_c, lr_h)
            #lr_preds = tf.nn.max_pool(lr_h, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME')
            lr_preds = lr_h
        return new_state, lr_preds

    def _run_hr_cell(self, state, inputs, keep_prob, scope='DR_CONV_LSTM', mode='train'):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            if mode == 'train':
                state = (tf.nn.dropout(state[0], keep_prob), tf.nn.dropout(state[1], keep_prob))
            hr_c, hr_h = self._hr_cell(inputs, state=state, scope=scope+'_hr')
            new_state = (hr_c, hr_h)
            hr_preds = hr_h
        return new_state, hr_preds

    def _run_inhibation_cell(self, state, inputs, keep_prob, scope='DR_CONV_LSTM', mode='train'):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            if mode == 'train':
                state = (tf.nn.dropout(state[0], keep_prob), tf.nn.dropout(state[1], keep_prob))
            hr_c, hr_h = self._inhibation_cell(inputs, state=state, scope=scope+'_hr')
            new_state = (hr_c, hr_h)
            hr_preds = hr_h
        return new_state, hr_preds
