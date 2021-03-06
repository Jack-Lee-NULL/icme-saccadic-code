#
#
# file: TrainModeD.py
# author: Jingquan Lee
# date: 2019-03-13
#
#

from train.TrainModeB import TrainModeB
from model.DResConvLSTMA import DResConvLSTMA

class TrainModeD(TrainModeB):

    def _init_model(self):
        self._preds = DResConvLSTMA(filter_size=self._filter_size,
                inputs_channel=self._inputs_channel, shape=self._shape,
                c_h_channel=self._c_h_channel, forget_bias=self._forget_bias,
                num_steps=self._num_steps, batch_size=self._batch_size)
