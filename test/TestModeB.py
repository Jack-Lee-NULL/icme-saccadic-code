#
#
# file: TestModeB.py
# author: Jingquan Lee
# date: 2019-03-16
#
#

import os

import numpy as np
import cv2

from test.TestModeA import TestModeA
from model.DResConvLSTM import DResConvLSTM

class TestModeB(TestModeA):

    def _init_model(self):
        self._predictor = DResConvLSTM(filter_size=self._filter_size,
                inputs_channel=self._inputs_channel, shape=self._shape,
                c_h_channel=self._c_h_channel, forget_bias=self._forget_bias,
                num_steps=self._num_steps)

    def _decode_preds(self, predicts):
        lr_preds = predicts[0] 
        hr_preds = predicts[1]
        lr_preds[:, :, 0] = lr_preds[:, :, 0] * self._shape[1]
        lr_preds[:, :, 1] = lr_preds[:, :, 1] * self._shape[0]
        hr_preds[:, :, 0] = hr_preds[:, :, 0] * self._shape[3]
        hr_preds[:, :, 1] = hr_preds[:, :, 1] * self._shape[2]
        lr_preds = np.around(lr_preds, 0)
        hr_preds = np.around(hr_preds, 0)
        preds = np.zeros(np.shape(lr_preds))
        preds[:, :, 0] = lr_preds[:, :, 0] * self._shape[3] + hr_preds[:, :, 0]
        preds[:, :, 1] = lr_preds[:, :, 1] * self._shape[2] + hr_preds[:, :, 1]
        preds = preds.astype('int32')
        print(preds.shape)
        return preds

    def __get_h_init(self, shape, coord, kernel_size):
        """Generate initial hidden state by using Gaussian kernel
        blur initial coordination, sigma is -1.
        Args:
            -shape: a tuple of (int, int), shape of hidden state
            -coord: a tuple of (float, float), normalized coordination
            -kernel_size: a int, Gaussian kernel size
        Returns:
            -init: a Tensor, generated initial hidden state
        """
        init = np.zeros(shape)
        coord = (np.around(coord[0] * shape[1]), np.around(coord[1] * shape[0]))
        init[int(coord[1])][int(coord[0])] = 1.0
        kernel = cv2.getGaussianKernel(kernel_size, sigma=-1)
        init = cv2.filter2D(init, ddepth = -1, kernel=kernel)
        init = cv2.filter2D(init, ddepth = -1, kernel=kernel.T)
        init = init[:, :, np.newaxis]
        return init

    def _generate_feed_dict(self, idxs):
        lr_features = []
        hr_features = []
        for idx in idxs:
            lr_feature = np.load(os.path.join(self._feature_dir[0], str(idx[0])+'.npy'))
            hr_feature = np.load(os.path.join(self._feature_dir[1], str(idx[0])+'.npy'))
            lr_features.append(lr_feature[:, :, :])
            hr_features.append(hr_feature[:, :, :, :])
        lr_features = np.array(lr_features)
        hr_features = np.array(hr_features)
        lr_h_init = []
        hr_h_init = []
        for idx in idxs:
            lr_h_init.append(self.__get_h_init(shape=(self._shape[0], self._shape[1]),
                    coord=(0.5, 0.5), kernel_size=15))
            hr_h_init.append(self.__get_h_init(shape=(self._shape[2], self._shape[3]),
                    coord=(0.5, 0.5), kernel_size=7))
        lr_h_init = np.array(lr_h_init)
        hr_h_init = np.array(hr_h_init)
        lr_c_init = np.zeros((np.shape(idxs)[0], self._shape[0], self._shape[1], self._c_h_channel[0]))
        hr_c_init = np.zeros((np.shape(idxs)[0], self._shape[2], self._shape[3], self._c_h_channel[2]))
        feed_dict = {self._predictor.lr_h_init: lr_h_init, self._predictor.hr_h_init: hr_h_init,
                self._predictor.lr_c_init: lr_c_init, self._predictor.hr_c_init: hr_c_init,
                self._predictor.lr_inputs: lr_features, self._predictor.hr_inputs: hr_features}
        return feed_dict
