#
#
# file: DResConvLSTMCellA.py
# author: Jingquan Lee
# date: 2019-03-18
#
#

from model.DResConvLSTMCell import DResConvLSTMCell
from model.ModifiedConvLSTMCellA import ModifiedConvLSTMCellA

class DResConvLSTMCellA(DResConvLSTMCell):
    """Double resolution conv-lstm cell,
    basic cell is modified conv-lstm cell A
    """
    
    def _init_cell(self):
        self._lr_cell = ModifiedConvLSTMCellA(shape=self._shape_lr,
                filter_size=self._filter_size_lr,
                num_features=self._c_h_channel[0],
                bias_start=self._forget_bias[0])
        self._hr_cell = ModifiedConvLSTMCellA(shape=self._shape_hr,
                filter_size=self._filter_size_hr,
                num_features=self._c_h_channel[2],
                bias_start=sel._forget_bias[1])


