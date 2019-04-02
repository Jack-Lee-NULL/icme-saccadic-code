#
#
# file: TestModeF.py
# author: Jingquan Lee
# date: 2019-03-26
#

import os

import numpy as np
import cv2

from test.TestModeA import TestModeA
from model.SingleConvLSTMA import SingleConvLSTMA

class TestModeF(TestModeA):

    def _init_model(self):
        self._feature_dir = self._feature_dir.split('\n')
        self._predictor = SingleConvLSTMA(filter_size=self._filter_size,
                inputs_channel=self._inputs_channel, shape=self._shape,
                c_h_channel=self._c_h_channel, forget_bias=self._forget_bias,
                num_steps=self._num_steps)

    def _decode_preds(self, predicts):
        scanpath = []
        for i in range(self._batch_size):
            scanpath_img = []
            for j in range(self._num_steps):
                coord = np.argmax(predicts[i, j, :, :, :])
                y = coord // self._shape[1]
                x = coord % self._shape[1]
                scanpath_img.append([x, y])
            scanpath.append(scanpath_img)
        return scanpath

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
        features = []
        for idx in idxs:
            feature = np.load(os.path.join(self._feature_dir[0], str(idx[0])+'.npy'))
            features.append(feature)
        features = np.array(features)
        scanpaths = []
        h_init = []
        for idx in idxs:
            h_init.append(self.__get_h_init((self._shape[0], self._shape[1]),
                    coord=(0.3, 0.5), kernel_size=45))
        c_init = np.zeros((np.shape(idxs)[0], self._shape[0], self._shape[1], self._c_h_channel[0]))
        feed_dict = {self._predictor.c_init: c_init, self._predictor.h_init: h_init,
                self._predictor._inputs: features}
        return feed_dict
