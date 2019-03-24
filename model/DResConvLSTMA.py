#
#
# file: DResConvLSTMA.py
# author: Jingquan Lee
# date: 2019-03-18
#
#

from model.DResConvLSTM import DResConvLSTM
from model.DResConvLSTMCellA import DResConvLSTMCellA

class DResConvLSTMA(DResConvLSTM):
    """Double resolution conv-lstm, constructed with
    DResConvLSTMCellA
    """

    def _init_cell(self):
        self._cell = DResConvLSTMCellA(
                filter_size_lr=(self._filter_size[0], self._filter_size[1]),
                filter_size_hr=(self._filter_size[2], self._filter_size[3]),
                shape_lr=(self._shape[0], self._shape[1]),
                shape_hr=(self._shape[2], self._shape[3]),
                c_h_channel=self._c_h_channel,
                forget_bias=self._forget_bias,
                activation=self._activation,
                batch_size=self._batch_size)


