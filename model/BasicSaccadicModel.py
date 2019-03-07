#
# file: BasicSaccadicModel.py
# author: Jingquan Lee
# date: 2019-03-07
#
#

from BasicConvLSTMCell import BasicConvLSTMCell

class BasicSaccadicModel:

    def __init__(self, shape, filter_size, 
            num_features=2, forget_bias=1.0,
            activation=tf.nn.tanh):
        self._cell = BasicConvLSTMCell(shape, filter_size,
                num_features, forget_bias=forget_bias,
                activation=activation, state_is_tuple=True)

    def __call__(self, state, inputs):
        new_state = self._cell(inputs, state)
        return new_state
