#
#
# file: ModifiedConvLSTMCellA.py
# author: Jingquan Lee
# date: 2019-03-18
#
#

import tensorflow as tf

class ModifiedConvLSTMCellA:
    """Modified Conv-LSTM cell.
    """

    def __init__(self, shape, filter_size, num_features, h_depth=1):
        """Initialize Modified Conv-LSTM cell.
        Args:
            -shape: a tuple of int, the height and width of the inputs,
            cell_state, hidden_state.
            -filter_size: a tuple of int, the height and width of filter.
            -num_features: int, the num of channels of cell state.
            -h_depth: int, the num of channels of hidden state.
        """
        self._shape = shape
        self._filter_size = filter_size
        self._num_features = num_features
        self._h_depth = h_depth

    def __call__(self, inputs, state, scope="MConvLSTM", bias_start=0.0):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
           c, h = state
           f = tf.multiply(h, inputs)
           matrix1 = tf.get_variable(
                   'Matrix1', [self._filter_size[0], self._filter_size[1],
                   inputs.shape[3], self._num_features],
                   dtype=tf.float32)
           matrix2 = tf.get_variable(
                   'Matrix2', [self._filter_size[0], self._filter_size[1],
                   inputs.shape[3], self._num_features], dtype=tf.float32)
           matrix3 = tf.get_variable(
                   'Matrix3', [self._filter_size[0], self._filter_size[1], 
                   2 * self._num_features, self._h_depth], dtype=tf.float32)
           bias1 = tf.get_variable(
                   'Bias1', [self._num_features], dtype=tf.float32,
                   initializer=tf.constant_initializer(bias_start, dtype=tf.float32))
           bias2 = tf.get_variable(
                   'Bias2', [self._num_features], dtype=tf.float32,
                   initializer=tf.constant_initializer(bias_start, dtype=tf.float32))
           bias3 = tf.get_variable(
                   'Bias3', [self._h_depth], dtype=tf.float32,
                   initializer=tf.constant_initializer(bias_start, dtype=tf.float32))
        
           res1 = tf.nn.conv2d(f, matrix1, strides=[1, 1, 1, 1], padding='SAME')
           res2 = tf.nn.conv2d(f, matrix2, strides=[1, 1, 1, 1], padding='SAME')
           res1 = tf.tanh(res1 + bias1)
           res2 = tf.sigmoid(res2 + bias2)
           res1 = tf.multiply(h, c) + res1
           res = tf.concat([res1, res2], axis=3)
           c = res1
           h = tf.nn.conv2d(res, matrix3, strides=[1, 1, 1, 1], padding='SAME')
           h = tf.sigmoid(h + bias3)
           new_state = (c, h)
           return new_state

