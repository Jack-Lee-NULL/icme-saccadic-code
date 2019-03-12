#
#
# file: DResConvLSTM.py
# author: Jingquan Lee
# date: 2019-03-12
#
#

import tensorflow as tf

from BasicConvLSTMCell import BasicConvLSTMCell

class DResConvLSTM:
    """Double resolution conv-lstm
    """

    def __init__(self, filter_size, inputs_channel, shape=(48, 64)
            c_h_channel=(1, 1), forget_bias=(1.0, 1.0),
            activation=(tf.nn.tanh, tf.nn.tanh), num_steps=8):
        """Intialize double resolution conv-lstm model. lr(low resolution),
        hr(high resolution)
        Args:
           -shape: a tuple, (rows, cols), shape of inputs, hidden
           state and cell state of low resolution conv-lstm
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
        pass

    def __call__(self):
        """construct double resolution conv-lstm model
        Returns:
            -lr_preds: a list, predctions of low resolution
            img by lr conv-lstm
            -hr_preds: a list, predictions of local high resolution
            img
        """
        pass

