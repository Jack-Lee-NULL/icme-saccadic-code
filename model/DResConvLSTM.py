#
#
# file: DResConvLSTM.py
# author: Jingquan Lee
# date: 2019-03-12
#
#

import tensorflow as tf

from model.DResConvLSTMCell import DResConvLSTMCell

class DResConvLSTM:
    """Double resolution conv-lstm
    """

    def __init__(self, filter_size=(3, 3, 3, 3), inputs_channel=(2048, 64), shape=(48, 64, 16, 16),
            c_h_channel=(1, 1), forget_bias=(1.0, 1.0),
            activation=(tf.nn.tanh, tf.nn.tanh), num_steps=8):
        """Intialize double resolution conv-lstm model. lr(low resolution),
        hr(high resolution)
        Args:
           -shape: a tuple of 4 ints, (lr_rows, lr_cols, hr_rows, hr_cols), 
           shape of inputs, hidden state and cell state of lr and hr
           -filter_size: a tuple, (lr filter size, hr filter size)
           both of lr filter size and hr filter size are tuple (row, col)
           -inputs_channel: a tuple, the number of inputs channel of
           (lr, hr)
           -c_h_channel: a tuple, the number of channels of hidden 
           and cell state of hr and lr.
           -forget_bias: a tuple, (float, float), the bias added 
           to forget gates of lr and hr
           -activation: a tuple, Activation function of the inner states
           of lr and hr
           -num_steps: int, the number of cells
        """
        self._num_steps = num_steps
        self.lr_c_init = tf.placeholder(tf.float32,
                (None, shape[0], shape[1], c_h_channel[0]), name='lr_c_init')
        self.hr_c_init = tf.placeholder(tf.float32,
                (None, shape[2], shape[3], c_h_channel[1]), name='hr_c_init')
        self.lr_h_init = tf.placeholder(tf.float32,
                (None, shape[0], shape[1], c_h_channel[0]), name='lr_h_init')
        self.hr_h_init = tf.placeholder(tf.float32,
                (None, shape[2], shape[3], c_h_channel[1]), name='hr_h_init')
        self.lr_inputs = tf.placeholder(tf.float32,
                (None, shape[0], shape[1], inputs_channel[0]), name='lr_inputs')
        self.hr_inputs = tf.placeholder(tf.float32,
                (None, shape[0] * shape[1], shape[2], shape[3], inputs_channel[1]), name='hr_inputs')
        self._cell = DResConvLSTMCell(
                filter_size_lr=(filter_size[0], filter_size[1]),
                filter_size_hr=(filter_size[2], filter_size[3]),
                shape_lr=(shape[0], shape[1]),
                shape_hr=(shape[2], shape[3]),
                c_h_channel=c_h_channel,
                forget_bias=forget_bias,
                activation=activation)
        """
        self._cell = []
        for _ in range(num_steps):
            self._cell.append(DResConvLSTMCell(
                    filter_size_lr=(filter_size[0], filter_size[1]),
                    filter_size_hr=(filter_size[2], filter_size[3]),
                    inputs_channel=inputs_channel,
                    shape_lr=(shape[0], shape[1]),
                    shape_hr=(shape[2], shape[3]),
                    c_h_channel=c_h_channel,
                    forget_bias=forget_bias,
                    activation=activation))

        """

    def __call__(self):
        """construct double resolution conv-lstm model
        Returns:
            -lr_preds: a list, predctions of low resolution
            img by lr conv-lstm
            -hr_preds: a list, predictions of local high resolution
            img
        """
        lr_preds = []
        hr_preds = []
        for i in range(self._num_steps):
            scope = 'DRes_Conv_LSTM' #+ str(i)
            if i == 0:
                lr_c = self.lr_c_init
                lr_h = self.lr_h_init
                hr_c = self.hr_c_init
                hr_h = self.hr_h_init
                state = (lr_c, lr_h, hr_c, hr_h)
            inputs = (self.lr_inputs[:, :, :, :],
                    self.hr_inputs[:, i, :, :, :])
            state, lr_pred, hr_pred = self._cell(
                    state=state, inputs=inputs, scope=scope)
            lr_preds.append(lr_pred)
            hr_preds.append(hr_pred)
        lr_preds = tf.concat(lr_preds, axis=1)
        hr_preds = tf.concat(hr_preds, axis=1)
        return lr_preds, hr_preds

