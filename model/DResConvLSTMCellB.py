#
#
# file: DResConvLSTMCellB.py
# author: Jingquan Lee
# date: 2019-03-24
#
#

from model.DResConvLSTMCell import DResConvLSTMCell
from model.ModifiedConvLSTMCellB import ModifiedConvLSTMCellB

class DResConvLSTMCellB(DResConvLSTMCell):

    def _init_cell(self):
        self._lr_cell = ModifiedConvLSTMCellB(shape=self._shape_lr,
                filter_size=self._filter_size_lr,
                num_features=self._c_h_channel[0])
        self._hr_cell = ModifiedConvLSTMCellB(shape=self._shape_hr,
                filter_size=self._filter_size_hr,
                num_features=self._c_h_channel[2])
