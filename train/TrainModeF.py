#
#
# file: TrainModeF.py
# author: Jingquan Lee
# date: 2019-03-26
#
#

import os

import numpy as np
import cv2
import tensorflow as tf


from train.TrainModeB import TrainModeB
from model.SingleConvLSTMA import SingleConvLSTMA

class TrainModeF(TrainModeB):

    def _init_model(self):
        self._feature_dir = self._feature_dir.split('\n')
        self._preds = SingleConvLSTMA(filter_size=self._filter_size,
                inputs_channel=self._inputs_channel, shape=self._shape,
                c_h_channel=self._c_h_channel, forget_bias=self._forget_bias,
                num_steps=self._num_steps)

    def _init_holder(self):
        self._labels_holder = tf.placeholder(name='labels', 
                shape=(None, self._num_steps, self._shape[0], self._shape[1], self._c_h_channel[1]),
                dtype=tf.float32)

    def _compute_loss(self):
        preds = self._preds()
        labels = self._labels_holder
        loss = 0.0
        weight = (self._labels_holder >= 0)
        weight = tf.cast(weight, dtype=tf.float32)
        #preds = tf.multiply(weight, preds)
        #labels = tf.multiply(self._labels_holder, weight)
        loss = (- tf.multiply(labels, tf.log(preds)) -
                tf.multiply((1 - labels), tf.log(1 - preds)))
        loss = tf.multiply(loss, weight)
        loss = tf.reduce_sum(loss)
        loss = loss / self._batch_size
        return loss

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

    def _generate_feed_dict(self, idxs):
        features = []
        for idx in idxs:
            feature = np.load(os.path.join(self._feature_dir[0], str(idx[0])+'.npy'))
            features.append(feature)
        features = np.array(features)
        scanpaths = []
        h_init = []
        for idx in idxs:
            s = np.load(os.path.join(self._feature_dir[1], str(idx[0])+'.npy'))
            scanpath = s[idx[1]][1: 9, :, :, np.newaxis]
            h_init.append(s[idx[1]][0, :, :, np.newaxis])
            scanpaths.append(scanpath)
        c_init = np.zeros((np.shape(idxs)[0], self._shape[0], self._shape[1], self._c_h_channel[0]))
        feed_dict = {self._preds.c_init: c_init, self._preds.h_init: h_init,
                self._preds._inputs: features, self._labels_holder: scanpaths}
        return feed_dict



