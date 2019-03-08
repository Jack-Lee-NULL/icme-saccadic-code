#
# file: BasicSaccadicModel.py
# author: Jingquan Lee
# date: 2019-03-07
#
#

import tensorflow as tf

from BasicConvLSTMCell import BasicConvLSTMCell

class BasicSaccadicModel:
    """A basic model to predict scanpath, just use basic conv-lstm
    """

    def __init__(self, shape, filter_size, inputs_channel=64,
            c_h_channel=1, forget_bias=1.0,
            activation=tf.nn.tanh, num_steps=8):
        """Intialize basic saccadic model
        Args:
            -shape: a (int, int) tuple, inputs, hidden state, cell state.
            -filter_size: a (int, int) tuple, that is the height and width
            of filters in cell.
            -inputs_channel: int, the number of channels of inputs.
            -c_h_channel: int, the number of channels of hidden and cell state.
            -forget_bias: float, the bias added to forget gates.
            -activation: Activation function of the inner states
            -num_steps: int, the number of cells.
        """
        self._cell = BasicConvLSTMCell(shape, filter_size,
                num_features=c_h_channel, forget_bias=forget_bias,
                activation=activation, state_is_tuple=True)
        self._num_steps = num_steps
        self._c_init = tf.placeholder(tf.float32, 
                (None, shape[0], shape[1], c_h_channel), name='c_init')
        self._h_init = tf.placeholder(tf.float32, 
                (None, shape[0], shape[1], c_h_channel), name='h_init')
        self._inputs = tf.placeholder(tf.float32, 
                (None, shape[0], shape[1], inputs_channel), name='inputs')

    @property
    def c_init(self):
        return self._c_init

    @property
    def h_init(self):
        return self._h_init

    @property
    def inputs(self):
        return self._inputs

    def __call__(self):
        """construct basic saccadic model
        Returns:
            -preds: a list, output of this model
        """
        preds = []
        for i in range(self._num_step):
            if i == 0:
               c = self._c_init
               h = self._h_init
            inputs = self._inputs
            (c, h) = self._cell(inputs, state=(c, h), scope='BSM')
            h_flatten = tf.layers.flatten(h)
            with tf.varaiable_scope('BSM', reuse=tf.AUTO_REUSE)
                w = tf.get_variable('ouput_w', shape=(h_flatten.shape[0], 2)
                        dtype=tf.float32)
                b = tf.get_variable('output_b', shape=(1, 2), dtype=tf.float32)
            output = tf.matmul(h_flatten, w) + b
            preds.append(output)
        return preds
