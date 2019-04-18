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
import keras.backend as K
from keras_applications.imagenet_utils import preprocess_input


from train.TrainModeB import TrainModeB
from model.SingleConvLSTMC import SingleConvLSTMC

class TrainModeF(TrainModeB):

    def _init_model(self):
        self._feature_dir = self._feature_dir.split('\n')
        self._preds = SingleConvLSTMC(filter_size=self._filter_size,
                inputs_channel=self._inputs_channel, shape=self._shape,
                c_h_channel=self._c_h_channel, forget_bias=self._forget_bias,
                num_steps=self._num_steps, batch_size=self._batch_size)

    def _init_holder(self):
        self._labels_holder = tf.placeholder(name='labels', 
                shape=(None, self._num_steps, self._shape[0], self._shape[1], 1),
                dtype=tf.float32)

    def _compute_loss(self):
        preds = self._preds()[0]
        labels = self._labels_holder
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=preds)
        loss = loss / self._batch_size / self._num_steps
        loss = tf.reduce_sum(loss)
        """
        preds = self._preds(mode='train')[0]
        labels = self._labels_holder
        loss = 0.0
        weight = labels > 0
        weight = tf.cast(weight, dtype=tf.float32)
        preds = tf.multiply(preds, weight)
        loss = tf.losses.mean_squared_error(labels, preds)
        loss = loss * self._num_steps
        """
        return loss

    def _decode_predicts(self, predicts):
        scanpath = []
        predicts = np.array(predicts)
        for i in range(self._batch_size):
            scanpath_img = []
            for j in range(self._num_steps):
                coord = np.argmax(predicts[i, j, :, :, :])
                y = coord // self._shape[1]
                x = coord % self._shape[1]
                scanpath_img.append([x, y])
            scanpath.append(scanpath_img)
        scanpath = np.array(scanpath)
        """
        predicts = np.array(predicts)
        predicts[:, :, 0] = predicts[:, :, 0] * self._shape[1]
        predicts[:, :, 1] = predicts[:, :, 1] * self._shape[0]
        predicts = predicts.astype('int32')
        scanpath = np.concatenate([predicts[:, :, 0], predicts[:, :, 1]], axis=1)
        """
        return scanpath

    def _generate_feed_dict(self, idxs):
        features_lr = []
        features_hr = []
        for idx in idxs:
            feature = np.load(os.path.join(self._feature_dir[0], str(idx[0])+'.npy'))
            feature = preprocess_input(feature, backend=K)
            features_lr.append(feature)
            feature = np.load(os.path.join(self._feature_dir[1], str(idx[0])+'.npy'))
            feature = preprocess_input(feature, backend=K)
            features_hr.append(feature)
        features_lr = np.array(features_lr)
        features_hr = np.array(features_hr)

        scanpaths = []
        o_init = []
        for idx in idxs:
            s = np.load(os.path.join(self._feature_dir[2], str(idx[0])+'.npy'))
            o_init.append(s[idx[1]][0: self._num_steps, :, :, np.newaxis])
            #scanpath = self._scanpath[idx[0]][idx[1]][1: self._num_steps+1, 0: 2]
            scanpaths.append(s[idx[1]][1: self._num_steps+1, :, :, np.newaxis])
        c_init = np.zeros((np.shape(idxs)[0], self._shape[2], self._shape[3], self._c_h_channel[0]))
        h_init = np.zeros((np.shape(idxs)[0], self._shape[2], self._shape[3], self._c_h_channel[1]))
        feed_dict = {self._preds.c_init: c_init, self._preds.h_init: h_init,
                self._preds.inputs[0]: features_lr,
                self._preds.inputs[1]: features_hr,
                self._labels_holder: scanpaths,
                self._preds.o_init: o_init}
        return feed_dict

