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
                   self._h_depth, 1], dtype=tf.float32, 
                   initializer=tf.orthogonal_initializer())
           matrix2 = tf.get_variable(
                   'Matrix2', [self._filter_size[0], self._filter_size[1],
                   self._h_depth, 1], dtype=tf.float32,
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
                   initializer=tf.constant_initializer(bias_start, dtype=tf.float32))
           bias2 = tf.get_variable(
                   'Bias2', [1], dtype=tf.float32,
                   initializer=tf.constant_initializer(bias_start, dtype=tf.float32))
           bias3 = tf.get_variable(
                   'Bias3', [self._num_features], dtype=tf.float32,
                   initializer=tf.constant_initializer(bias_start, dtype=tf.float32))
           bias4 = tf.get_variable(
                   'Bias4', [self._num_features], dtype=tf.float32,
                   initializer=tf.constant_initializer(bias_start, dtype=tf.float32))
           bias5 = tf.get_variable(
                   'Bias5', [self._h_depth], dtype=tf.float32,
                   initializer=tf.constant_initializer(bias_start, dtype=tf.float32))
           forget_gate = tf.nn.conv2d(h, matrix1, strides=[1, 1, 1, 1], padding='SAME')
           forget_gate = tf.sigmoid(forget_gate + bias1)
           region = tf.nn.conv2d(h, matrix2, strides=[1, 1, 1, 1], padding='SAME')
           region = tf.sigmoid(region + bias2)
           update_gate = tf.nn.conv2d(region * inputs, matrix3, strides=[1, 1, 1, 1],
                   padding='SAME')
           update_gate = tf.tanh(update_gate + bias3)
           origin_feature = tf.nn.conv2d(inputs, matrix3, strides=[1, 1, 1, 1], padding='SAME')
           origin_feature = tf.sigmoid(origin_feature + bias4)
           new_c = c  + update_gate
           residual = tf.concat([origin_feature, new_c], axis=3)
           new_h = tf.nn.conv2d(residual, matrix5, strides=[1, 1, 1, 1], padding='SAME')
           new_c = new_c * forget_gate
           new_h = tf.sigmoid(new_h + bias5)
           new_state = (new_c, new_h)
           return new_state
