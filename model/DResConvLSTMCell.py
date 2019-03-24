#
#
# file: DResConvLSTMCell.py
# author: Jingquan Lee
# date: 2019-03-12
#
#

import tensorflow as tf

from model.BasicConvLSTMCell import BasicConvLSTMCell

class DResConvLSTMCell:
    """Double resolution conv-lstm cell
    """

    def __init__(self, filter_size_lr, filter_size_hr, shape_lr=(48, 64),
            shape_hr=(16, 16), c_h_channel=(1, 1), forget_bias=(1.0, 1.0),
            activation=(tf.nn.tanh, tf.nn.tanh), batch_size=10):
        """Intialize double resolution conv-lstm cell. lr(low resolution),
        hr(high resolution)
        Args:
           -shape_lr: a tuple, (rows, cols), shape of inputs, hidden
           state and cell state of low resolution conv-lstm
           -shape_hr: a tuple, (rows, cols), shape of inputs, hidden
           state and cell state of high resolution conv-lstm
           -filter_size_lr: a tuple, (rows, cols), lr filter size
           -filter_size_hr:a tuple, (rows, cols), hr filter size
           -c_h_channel: a tuple, the number of channels of hidden 
           and cell state of hr and lr.
           -forget_bias: a tuple, (float, float), the bias added 
           to forget gates of lr and hr
           -activation: a tuple, Activation function of the inner states
           of lr and hr
        """
        self._batch_size = batch_size
        self._shape_lr = shape_lr
        self._shape_hr = shape_hr
        self._filter_size_lr = filter_size_lr
        self._filter_size_hr = filter_size_hr
        self._c_h_channel = c_h_channel
        self._forget_bias = forget_bias
        self._activation = activation
        self._pool_strides = (1, shape_lr[0] / shape_hr[0], shape_lr[1] / shape_hr[1], 1)
        self._pool_ksize = (shape_lr[0] / shape_hr[0] + 1,
                shape_lr[1]/ shape_hr[1] + 1)
        self._init_cell()

    def _init_cell(self):
        self._lr_cell = BasicConvLSTMCell(self._shape_lr, self._filter_size_lr,
                num_features=self._c_h_channel[0], forget_bias=self._forget_bias[0],
                activation=self._activation[0], state_is_tuple=True)
        self._hr_cell = BasicConvLSTMCell(self._shape_hr, self._filter_size_hr,
                num_features=self._c_h_channel[1], forget_bias=self._forget_bias[1],
                activation=self._activation[1], state_is_tuple=True)

    def __call__(self, state, inputs, scope='DR_CONV_LSTM', keep_prob=0.8):
        """construct double resolution conv-lstm cell
        Args:
            -state: a tuple, (lr_c, lr_h, hr_c, hr_h)
            -inputs: a tuple, (Tensor, Tensor)
        Returns:
            -new_state: a tuple of 4 Tensors, new state of this cell,
            (lr_c, lr_h, hr_c, hr_h)
            -lr_preds: a float, in (0, 1), prediction of low resolution
            img by lr conv-lstm cell
            -hr_preds: a float, in (0, 1), prediction of high resolution
            img by lr conv-lstm cell
        """
        lr_inputs = tf.nn.dropout(inputs[0], keep_prob)
        hr_inputs = tf.nn.dropout(inputs[1], keep_prob)

        lr_state, lr_preds = self._run_lr_cell(
                state=(state[0], state[1]), inputs=lr_inputs,
                keep_prob=keep_prob, scope=scope)
        hr_input_c = tf.nn.max_pool(lr_state[0], 
                ksize=(1, self._pool_ksize[0], self._pool_ksize[1], 1),
                strides=self._pool_strides, padding='SAME')
        region_idx = self._get_region_id(lr_preds[:, 0, :])
        hr_input = []
        for i in range(self._batch_size):
            hr_input.append(hr_inputs[i, region_idx[i], :, :, :])
        hr_input = tf.concat([hr_input, hr_input_c], axis=3)
        hr_state, hr_preds = self._run_hr_cell(
                state=(state[2], state[3]), inputs=hr_input,
                keep_prob=keep_prob, scope=scope)
        new_state = (lr_state[0], lr_state[1], hr_state[0], hr_state[1])
        return new_state, lr_preds, hr_preds

    def _get_region_id(self, coord):
        x = tf.round(coord[:, 0] * self._shape_lr[1])
        y = tf.round(coord[:, 1] * self._shape_lr[0])
        idx = y * self._shape_lr[1] + x
        idx = tf.cast(idx, tf.int32)
        idx = tf.clip_by_value(idx, 0, self._shape_lr[1]*self._shape_hr[0]-1)
        return idx

    def _run_lr_cell(self, state, inputs, keep_prob, scope='DR_CONV_LSTM'):
        state = (tf.nn.dropout(state[0], keep_prob), tf.nn.dropout(state[1], keep_prob))
        lr_c, lr_h = self._lr_cell(inputs, state=state, scope=scope+'_lr')
        lr_h_flatten = tf.layers.flatten(lr_h)
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            w = tf.get_variable('lr_ouput_w', shape=(lr_h_flatten.shape[1], 2),
                    dtype=tf.float32)
            b = tf.get_variable('lr_output_b', shape=(1, 2), dtype=tf.float32)
        lr_preds = tf.matmul(lr_h_flatten, w) + b
        lr_preds = tf.expand_dims(lr_preds, axis=1)
        lr_preds = tf.nn.relu(lr_preds)
        new_state = (lr_c, lr_h)
        preds = lr_preds
        return new_state, preds

    def _run_hr_cell(self, state, inputs, keep_prob, scope='DR_CONV_LSTM'):
        state = (tf.nn.dropout(state[0], keep_prob), tf.nn.dropout(state[1], keep_prob))
        hr_c, hr_h = self._hr_cell(inputs, state=state, scope=scope+'_hr')
        hr_h_flatten = tf.layers.flatten(hr_h)
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            w = tf.get_variable('hr_output_w', shape=(hr_h_flatten.shape[1], 2),
                    dtype = tf.float32)
            b = tf.get_variable('hr_output_b', shape=(1, 2), dtype=tf.float32)
        hr_preds = tf.matmul(hr_h_flatten, w) + b
        hr_preds = tf.expand_dims(hr_preds, axis=1)
        hr_preds = tf.nn.relu(hr_preds)       
        new_state = (hr_c, hr_h)
        preds = hr_preds
        return new_state, preds
