#
#
# file: TestModeF.py
# author: Jingquan Lee
# date: 2019-03-26
#

import os

import numpy as np
import cv2
import tensorflow as tf
import keras.backend as K
from keras_applications.imagenet_utils import preprocess_input

from test.TestModeA import TestModeA
from ResNet50.resnet_part import ResNet50
from model.SingleConvLSTMC import SingleConvLSTMC

class TestModeF(TestModeA):

    def _init_model(self):
        self._feature_dir = self._feature_dir.split('\n')
        self._predictor = SingleConvLSTMC(filter_size=self._filter_size,
                batch_size=self._batch_size,
                inputs_channel=self._inputs_channel, shape=self._shape,
                c_h_channel=self._c_h_channel, forget_bias=self._forget_bias,
                num_steps=1)
        self._resnet = ResNet50(include_top=False)

    def _decode_preds(self, predicts):
        scanpath = []
        predicts = np.array(predicts)
        for i in range(np.shape(predicts)[0]):
            scanpath_img = []
            for j in range(self._num_steps):
                coord = np.argmax(predicts[i, j, :, :, :])
                y = coord // self._shape[1]
                x = coord % self._shape[1]
                scanpath_img.append([x, y])
            scanpath.append(scanpath_img)
        return scanpath

    def __get_h_init(self, shape, coord, kernel_size):
        init = np.zeros(shape)
        coord = (coord[0]*shape[1], coord[1]*shape[0])
        init[int(coord[1])][int(coord[0])] = 1.0
        kernel = cv2.getGaussianKernel(kernel_size, sigma=-1)
        init = cv2.filter2D(init, ddepth = -1, kernel=kernel)
        init = cv2.filter2D(init, ddepth = -1, kernel=kernel.T)
        init = init / np.max(init)
        init = init[:, :, np.newaxis]
        return init

    def __get_o(self, shape, o_pre, o, kernel_size):
        o_pre = np.array(o_pre[0, :, :, :, :])
        init = np.zeros(shape)
        coord = np.argmax(o)
        coord_y = coord // self._shape[1]
        coord_x = coord % self._shape[1]
        init[coord_y][coord_x] = 1.0
        kernel = cv2.getGaussianKernel(kernel_size, sigma=-1)
        init = cv2.filter2D(init, ddepth = -1, kernel=kernel)
        init = cv2.filter2D(init, ddepth = -1, kernel=kernel.T)
        init = init / np.max(init)
        init = init[np.newaxis, :, :, np.newaxis]
        return init   

    def _generate_feed_dict(self, idxs):
        features_lr = []
        features_hr = []
        blur_img = []
        for idx in idxs:
            feature = np.load(os.path.join(self._feature_dir[0], str(idx[0])+'.npy'))
            feature = preprocess_input(feature[np.newaxis, :, :, :], backend=K)
            blur_img.append(feature[0, :, :, :])
            feature = self._resnet.predict(feature)[0, :, :, :]
            features_lr.append(feature)
            feature = np.load(os.path.join(self._feature_dir[1], str(idx[0])+'.npy'))
            feature = preprocess_input(feature[np.newaxis, :, :, :], backend=K)
            feature = self._resnet.predict(feature)[0, :, :, :]
            features_hr.append(feature)
        features_lr = np.array(features_lr)
        features_hr = np.array(features_hr)
        blur_img = np.array(blur_img)

        o_init = []
        self._o_init = np.array(self._o_init)
        for i in range(np.shape(idxs)[0]):
            o = cv2.resize(self._o_init[i, :, :, :], (self._shape[3], self._shape[2]))[np.newaxis, :, :, np.newaxis]
            o_init.append(o)
        o_init = np.array(o_init)
        

        """
        for idx in idxs:
            o_init.append(self.__get_h_init((self._shape[0], self._shape[1]),
                    coord=(0.5, 0.5), kernel_size=150))
        """
        c_init = np.zeros((np.shape(idxs)[0], self._shape[2], self._shape[3], self._c_h_channel[0]))
        h_init = np.zeros((np.shape(idxs)[0], self._shape[2], self._shape[3], self._c_h_channel[1]))
        feed_dict = {self._predictor.c_init: c_init, self._predictor.h_init: h_init,
                self._predictor._inputs[0]: features_lr,
                self._predictor._inputs[1]: features_hr,
                self._predictor._inputs[2]: blur_img,
                self._predictor._o_init: o_init}
        return feed_dict

    def predicts(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        n_iters = np.shape(self._idxs)[0]
        predictor = self._predictor(mode='test')
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self._trained_model)
            preds = []
            for i in range(n_iters):
                idxs = self._idxs[i : i + 1, :]
                self._o_init = [self.__get_h_init((self._shape[0], self._shape[1]),
                    coord=(0.5, 0.5), kernel_size=49)]
                feed_dict = self._generate_feed_dict(idxs)
                pred = []
                for _ in range(self._num_steps):
                    o, c, h = sess.run(predictor, feed_dict)
                    pred.append(o)
                    o = self.__get_o((self._shape[0], self._shape[1]), feed_dict[self._predictor.o_init], o, 49)
                    self._o_init = o
                    feed_dict = self._generate_feed_dict(idxs)
                    feed_dict[self._predictor.c_init] = c
                    feed_dict[self._predictor.h_init] = h
                pred = np.concatenate(pred, axis=1)
                preds.append(pred)
        preds = np.concatenate(preds, axis=0)
        preds = self._decode_preds(preds)
        np.save(self._preds_path, preds)
        print ("Predictions have been saved to", self._preds_path)
