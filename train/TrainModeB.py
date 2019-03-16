#
#
# file: TrainModeB.py
# author: Jingquan Lee
# date: 2019-03-13
#
#

import os

import numpy as np
import cv2
import tensorflow as tf

from model.DResConvLSTM import DResConvLSTM

from train.TrainModeA import TrainModeA

class TrainModeB(TrainModeA):
    def __init__(self, learning_rate=0.0005, epochs=20, batch_size=10, shape=(48, 64, 16, 16),
                 print_every=1, save_every=1, log_path=None, filter_size=(3, 3, 3, 3),
                 inputs_channel=(2048, 64), c_h_channel=(1, 1), forget_bias=(1.0, 1.0),
                 save_model_path=None, pretrained_model=None, feature_dir=(None, None),
                 scanpath=(None, None), idxs=None, num_steps=8, num_validation=None):
        """Intialize TrainModeB which is extended from TrainModeA, lr indicates low 
        resolution lstm and hr indicates high resolution lstm
        Args(different from TrainModeA):
            -shape: a tuple of 4 ints, (lr rows, lr cols, hr rows, hr cols)
            -filter_size: a tuple of 4 ints, (lr rows, lr cols, hr rows, hr cols) of
            filter size in lr or hr
            -inputs_channel: a tuple of 2 ints, (lr input channels, hr input channels)
            -c_h_channel: a tuple of 2 ints, (lr channels, hr channels)
            -forget_bias: a tuple of 2 ints, (lr forget bias, hr forget bias)
            -init_hidden: a tuple of 2 Tensors, (lr hidden initial tensor,
            hr hidden initial Tensor)
            -scanpath: a tuple of 2 Tensors, (lr gt scanpath, hr gt scanpath)
        """
        super().__init__(learning_rate=learning_rate, epochs=epochs, batch_size=batch_size,
                shape=shape, print_every=print_every, save_every=save_every, log_path=log_path,
                filter_size=filter_size, inputs_channel=inputs_channel, c_h_channel=c_h_channel,
                forget_bias=forget_bias, init_hidden=(None, None), save_model_path=save_model_path,
                pretrained_model=pretrained_model, feature_dir=feature_dir, scanpath=scanpath,
                idxs=idxs, num_steps=num_steps, num_validation=num_validation)

    def _init_model(self):
        self._preds = DResConvLSTM(filter_size=self._filter_size,
                inputs_channel=self._inputs_channel, shape=self._shape,
                c_h_channel=self._c_h_channel, forget_bias=self._forget_bias,
                num_steps=self._num_steps)

    def _init_holder(self):
        self._lr_labels_holder = tf.placeholder(name='lr_labels', shape=(None, self._num_steps, 2), dtype=tf.float32)
        self._hr_labels_holder = tf.placeholder(name='hr_labels', shape=(None, self._num_steps, 2), dtype=tf.float32)
        self._labels_holder=tf.placeholder(name='labels', shape=(2, None, self._num_steps, 2),
                dtype=tf.float32)

    def _compute_loss(self):
        preds = self._preds()
        lr_labels = self._lr_labels_holder
        hr_labels = self._hr_labels_holder
        loss = 0.0
        weight = lr_labels > 0
        weight = tf.cast(weight, dtype=tf.float32)
        lr_preds = tf.multiply(preds[0], weight)
        weight = hr_labels > 0
        weight = tf.cast(weight, dtype=tf.float32)
        hr_preds = tf.multiply(preds[1], weight)
        loss += tf.losses.mean_squared_error(lr_labels, lr_preds)
        loss += tf.losses.mean_squared_error(hr_labels, hr_preds)
        return loss
        
    def _decode_predicts(self, predicts):
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
        preds = np.concatenate([preds[:, :, 0], preds[:, :, 1]], axis=1)
        return preds
        
    #def __get_hr_feature_idx(self, coord):
    #    """Gnerate the idx in hr region feature, for sake of
    #    drawing regions of feature as a list.
    #    """
    #    x = np.around(coord[:, 0] * self._shape[1], 0)
    #    y = np.around(coord[:, 1] * self._shape[0], 0)
    #    idx = y * self._shape[1] + x
    #    idx = idx.astype('int32')
    #    return idx

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
#            region_idx = self.__get_hr_feature_idx(self._scanpath[0][idx[0]][idx[1]][0: 8, :])
            lr_features.append(lr_feature[:, :, :])
            hr_features.append(hr_feature[:, :, :, :])
        lr_features = np.array(lr_features)
        hr_features = np.array(hr_features)
        lr_scanpaths = []
        hr_scanpaths = []
        for idx in idxs:
            lr_scanpath = self._scanpath[0][idx[0]][idx[1]][1: 9, 0: 2]
            lr_scanpaths.append(lr_scanpath)
            hr_scanpath = self._scanpath[1][idx[0]][idx[1]][1: 9, 0: 2]
            hr_scanpaths.append(hr_scanpath)
        lr_scanpaths = np.array(lr_scanpaths)
        hr_scanpaths = np.array(hr_scanpaths)
        lr_h_init = []
        hr_h_init = []
        for idx in idxs:
            lr_h_init.append(self.__get_h_init(shape=(self._shape[0], self._shape[1]),
                    coord=(self._scanpath[0][idx[0]][idx[1]][0, 0],
                           self._scanpath[0][idx[0]][idx[1]][0, 1]),
                    kernel_size=15))
            hr_h_init.append(self.__get_h_init(shape=(self._shape[2], self._shape[3]),
                    coord=(self._scanpath[1][idx[0]][idx[1]][0, 0],
                           self._scanpath[1][idx[0]][idx[1]][0, 1]),
                    kernel_size=7))
        lr_h_init = np.array(lr_h_init)
        hr_h_init = np.array(hr_h_init)
        lr_c_init = np.zeros((np.shape(idxs)[0], self._shape[0], self._shape[1], 1))
        hr_c_init = np.zeros((np.shape(idxs)[0], self._shape[2], self._shape[3], 1))
        feed_dict = {self._lr_labels_holder: lr_scanpaths, self._hr_labels_holder: hr_scanpaths,
                self._preds.lr_h_init: lr_h_init, self._preds.hr_h_init: hr_h_init,
                self._preds.lr_c_init: lr_c_init, self._preds.hr_c_init: hr_c_init,
                self._preds.lr_inputs: lr_features, self._preds.hr_inputs: hr_features,
                self._labels_holder: [lr_scanpaths, hr_scanpaths]}
        return feed_dict
