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
           interest = inputs
           c, h = state
           matrix1 = tf.get_variable(
                   'Matrix1', [self._filter_size[0], self._filter_size[1],
                   interest.shape[3]+self._h_depth, self._num_features], 
                   dtype=tf.float32)
           matrix2 = tf.get_variable(
                   'Matrix2', [self._filter_size[0], self._filter_size[1],
                   interest.shape[3]+self._h_depth, self._num_features], 
                   dtype=tf.float32)
           matrix3 = tf.get_variable(
                   'Matrix3', [self._filter_size[0], self._filter_size[1],
                   interest.shape[3]+self._h_depth, self._num_features],
                   dtype=tf.float32)
           matrix4 = tf.get_variable(
                   'Matrix4', [self._filter_size[0], self._filter_size[1],
                   self._num_features, self._num_features],
                   dtype=tf.float32)
           matrix5 = tf.get_variable(
                   'Matrix5', [self._filter_size[0], self._filter_size[1],
                   interest.shape[3]+self._h_depth, self._num_features],
                   dtype=tf.float32)
           bias1 = tf.get_variable(
                   'Bias1', [self._num_features], dtype=tf.float32,
                   initializer=tf.constant_initializer(self._bias_start, dtype=tf.float32))
           bias2 = tf.get_variable(
                   'Bias2', [self._num_features], dtype=tf.float32,
                   initializer=tf.constant_initializer(self._bias_start, dtype=tf.float32))
           bias3 = tf.get_variable(
                   'Bias3', [self._num_features], dtype=tf.float32,
                   initializer=tf.constant_initializer(self._bias_start, dtype=tf.float32))
           bias4 = tf.get_variable(
                   'Bias4', [self._num_features], dtype=tf.float32,
                   initializer=tf.constant_initializer(self._bias_start, dtype=tf.float32))
           bias5 = tf.get_variable(
                   'Bias5', [self._num_features], dtype=tf.float32,
                   initializer=tf.constant_initializer(self._bias_start, dtype=tf.float32))
           forget_gate = tf.concat([interest, h], axis=3)
           forget_gate = tf.nn.conv2d(forget_gate, matrix1, strides=[1, 1, 1, 1],
                   padding='SAME')
           forget_gate = tf.sigmoid(forget_gate + bias1)
           c = c * forget_gate

           update_gate = tf.concat([interest, h], axis=3)
           update_gate = tf.nn.conv2d(update_gate, matrix2, strides=[1, 1, 1, 1], 
                   padding='SAME')
           update_gate = tf.sigmoid(update_gate + bias2)

           gate_weight = tf.concat([interest, h], axis=3)
           gate_weight = tf.nn.conv2d(gate_weight, matrix3, strides=[1, 1, 1, 1], 
                   padding='SAME')
           gate_weight = tf.nn.tanh(gate_weight + bias3)
           update_gate = tf.multiply(update_gate, gate_weight)
           #update_gate = tf.clip_by_value(update_gate, 0, 1)
           #update_gate = tf.nn.conv2d(update_gate, matrix4, strides=[1, 1, 1, 1],
           #        padding='SAME')
           #update_gate = tf.nn.relu(update_gate + bias4)
           #new_c = tf.reduce_max([c, update_gate], axis=0)
           new_c = c + update_gate
           new_h = tf.concat([interest, h], axis=3)
           new_h = tf.nn.conv2d(new_h, matrix5, strides=[1, 1, 1, 1], padding='SAME')
           new_h = tf.sigmoid(new_h + bias5)
           new_h = tf.multiply(new_h, tf.tanh(new_c))
           new_state=(new_c, new_h)
           return new_state
