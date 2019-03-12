#
#
# file: DResConvLSTMCell.py
# author: Jingquan Lee
# date: 2019-03-12
#
#

class DResConvLSTMCell:
    """Double resolution conv-lstm cell
    """

    def __init__(self, filter_size, inputs_channel, shape_lr=(48, 64),
            shape_hr=(16, 16), c_h_channel(1, 1), forget_bias=(1.0, 1.0),
            activation=(tf.nn.tanh, tf.nn.tanh)):
        """Intialize double resolution conv-lstm cell. lr(low resolution),
        hr(high resolution)
        Args:
           -shape_lr: a tuple, (rows, cols), shape of inputs, hidden
           state and cell state of low resolution conv-lstm
           -shape_hr: a tuple, (rows, cols), shape of inputs, hidden
           state and cell state of high resolution conv-lstm
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
        """
        self._lr_cell = BasicConvLSTMCell(shape_lr, filter_size[0],
                num_features=c_h_channel[0], forget_bias=forget_bias[0],
                activation=activation[0], state_is_tuple=True)
        self._hr_cell = BasicConvLSTMCell(shape_hr, filter_size[1],
                num_features=c_h_channel[1], forget_bias=forget_bias[1],
                activation=activation[1], state_is_tuple=True)

    def __call__(self, state, inputs, scope='DR_CONV_LSTM'):
        """construct double resolution conv-lstm cell
        Args:
            -state: a tuple, (lr_c, lr_h, hr_c, hr_h)
            -inputs: a tuple, (Tensor, Tensor)
        Returns:
            -lr_preds: a float, in (0, 1), prediction of low resolution
            img by lr conv-lstm cell
            -hr_preds: a float, in (0, 1), prediction of high resolution
            img by lr conv-lstm cell
        """
        pass
