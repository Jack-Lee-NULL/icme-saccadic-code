#
#
# file: ModifiedConvLSTMCellB.py
# author: Jingquan Lee
# date: 2019-03-24
#
#

import tensorflow as tf

from model.ModifiedConvLSTMCellA import ModifiedConvLSTMCellA

class ModifiedConvLSTMCellB(ModifiedConvLSTMCellA):

    
    def __call__(self, inputs, state, scope="MConvLSTMB", bias_start=1.0):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
           x = inputs[0]
           o = inputs[1]
           c, h = state
           matrix1 = tf.get_variable(
                   'Matrix1', [self._filter_size[0], self._filter_size[1],
                   self._h_depth+x.shape[3]+self._num_features, 1], 
                   dtype=tf.float32, 
                   initializer=tf.orthogonal_initializer())
           matrix2 = tf.get_variable(
                   'Matrix2', [self._filter_size[0], self._filter_size[1],
                   self._h_depth+x.shape[3], 1], 
                   dtype=tf.float32, 
                   initializer=tf.orthogonal_initializer())
           matrix3 = tf.get_variable(
                   'Matrix3', [self._filter_size[0], self._filter_size[1],
                   self._h_depth+x.shape[3], self._num_features],
                   dtype=tf.float32,
                   initializer=tf.orthogonal_initializer())
           matrix4 = tf.get_variable(
                   'Matrix4', [self._filter_size[0], self._filter_size[1],
                   self._h_depth+x.shape[3]+self._num_features, self._h_depth],
                   dtype=tf.float32,
                   initializer=tf.orthogonal_initializer())
           bias1 = tf.get_variable(
                   'Bias1', [1], dtype=tf.float32,
                   initializer=tf.constant_initializer(self._bias_start, dtype=tf.float32))
           bias2 = tf.get_variable(
                   'Bias2', [self._num_features], dtype=tf.float32,
                   initializer=tf.constant_initializer(self._bias_start, dtype=tf.float32))
           bias3 = tf.get_variable(
                   'Bias3', [1], dtype=tf.float32,
                   initializer=tf.constant_initializer(self._bias_start, dtype=tf.float32))
           bias4 = tf.get_variable(
                   'Bias4', [self._h_depth], dtype=tf.float32,
                   initializer=tf.constant_initializer(self._bias_start, dtype=tf.float32))
           x_interest = tf.multiply(x, o)
           x_inhibation = x
           forget_gate = tf.concat([h, c, x_interest], axis=3)
           forget_gate = tf.nn.conv2d(forget_gate, matrix1, strides=[1, 1, 1, 1],
                   padding='SAME')
           forget_gate = tf.sigmoid(forget_gate + bias1)

           update_gate = tf.concat([h, x_interest], axis=3)
           update_gate = tf.nn.conv2d(update_gate, matrix2, strides=[1, 1, 1, 1], 
                   padding='SAME')
           update_gate = tf.tanh(update_gate + bias2)
           gate_weight = tf.concat([h, x_interest], axis=3)
           gate_weight = tf.nn.conv2d(gate_weight, matrix3, strides=[1, 1, 1, 1], 
                   padding='SAME')
           gate_weight = tf.sigmoid(gate_weight + bias3)

           new_c = c * forget_gate
           new_c = tf.reduce_max([new_c, update_gate * gate_weight], axis=0)
           new_h = tf.concat([new_c, h, x_inhibation], axis=3)
           new_h = tf.nn.conv2d(new_h, matrix4, strides=[1, 1, 1, 1], padding='SAME')
           new_h = tf.sigmoid(new_h + bias4)
           new_state=(new_c, new_h)
           return new_state
