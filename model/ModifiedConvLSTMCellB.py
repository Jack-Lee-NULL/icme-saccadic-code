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
           c, h = state
           matrix1 = tf.get_variable(
                   'Matrix1', [self._filter_size[0], self._filter_size[1],
                   self._h_depth+inputs.shape[3]+self._num_features, 1], 
                   dtype=tf.float32, 
                   initializer=tf.orthogonal_initializer())
           matrix2 = tf.get_variable(
                   'Matrix2', [self._filter_size[0], self._filter_size[1],
                   self._h_depth+inputs.shape[3]+self._num_features, 1], 
                   dtype=tf.float32, 
                   initializer=tf.orthogonal_initializer())
           matrix3 = tf.get_variable(
                   'Matrix3', [self._filter_size[0], self._filter_size[1],
                   inputs.shape[3], self._num_features], dtype=tf.float32,
                   initializer=tf.orthogonal_initializer())
           matrix4 = tf.get_variable(
                   'Matrix4', [self._filter_size[0], self._filter_size[1],
                   inputs.shape[3], self._num_features], dtype=tf.float32,
                   initializer=tf.orthogonal_initializer())
           matrix5 = tf.get_variable(
                   'Matrix5', [self._filter_size[0], self._filter_size[1],
                   2*self._num_features, self._h_depth], dtype=tf.float32,
                   initializer=tf.orthogonal_initializer())
           bias1 = tf.get_variable(
                   'Bias1', [1], dtype=tf.float32,
                   initializer=tf.constant_initializer(self._bias_start, dtype=tf.float32))
           bias2 = tf.get_variable(
                   'Bias2', [1], dtype=tf.float32,
                   initializer=tf.constant_initializer(self._bias_start, dtype=tf.float32))
           bias3 = tf.get_variable(
                   'Bias3', [self._num_features], dtype=tf.float32,
                   initializer=tf.constant_initializer(self._bias_start, dtype=tf.float32))
           bias4 = tf.get_variable(
                   'Bias4', [self._num_features], dtype=tf.float32,
                   initializer=tf.constant_initializer(self._bias_start, dtype=tf.float32))
           bias5 = tf.get_variable(
                   'Bias5', [self._h_depth], dtype=tf.float32,
                   initializer=tf.constant_initializer(self._bias_start, dtype=tf.float32))
           forget_gate = tf.concat([h, c, inputs], axis=3)
           forget_gate = tf.nn.conv2d(forget_gate, matrix1, strides=[1, 1, 1, 1], 
                   padding='SAME')
           forget_gate = tf.sigmoid(forget_gate + bias1)
           update_gate = tf.concat([h, c, inputs], axis=3)
           update_gate = tf.nn.conv2d(update_gate, matrix2, strides=[1, 1, 1, 1], 
                   padding='SAME')
           update_gate = tf.sigmoid(update_gate + bias2)
           update_gate = update_gate * inputs
           update_gate = tf.nn.conv2d(update_gate, matrix3, strides=[1, 1, 1, 1], 
                   padding='SAME')
           update_gate = tf.sigmoid(update_gate + bias3)
           origin_feature = tf.nn.conv2d(inputs, matrix4, strides=[1, 1, 1, 1], 
                   padding='SAME')
           origin_feature = tf.sigmoid(origin_feature + bias4)
           new_c = c * update_gate
           new_c = tf.reduce_max([new_c, update_gate], axis=0)
           new_h = tf.concat([new_c, origin_feature], axis=3)
           new_h = tf.nn.conv2d(new_h, matrix5, strides=[1, 1, 1, 1], padding='SAME')
           new_h = tf.sigmoid(new_h + bias5)
           new_state=(new_c, new_h)
           return new_state
