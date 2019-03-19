#
#
# file: TestModeD.py
# author: Jingquan Lee
# date: 2019-03-19
#
#

from test.TestModeB import TestModeB
from model.DResConvLSTMA import DResConvLSTMA

class TestModeD(TestModeB):

    def _init_model(self):
        self._predictor = DResConvLSTMA(filter_size=self._filter_size,
            inputs_channel=self._inputs_channel, shape=self._shape,
            c_h_channel=self._c_h_channel, forget_bias=self._forget_bias,
            num_steps=self._num_steps)
