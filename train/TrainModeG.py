#
#
# file: TrainModeG.py
# author: Jingquan Lee
# date: 2019-04-22
#
#
import os

import numpy as np
import cv2
import tensorflow as tf
import keras.backend as K
from keras_applications.imagenet_utils import preprocess_input

from model.DResConvLSTMC import DResConvLSTMC
from train.TrainModeF import TrainModeF
from ResNet50.resnet_part import ResNet50 as ResNet50_lr
from ResNet50.resnet_part_dilation import ResNet50 as ResNet50_hr

class TrainModeG(TrainModeF):
    def _init_model(self):
        self._feature_dir = self._feature_dir.split('\n')
        self._preds = DResConvLSTMC(filter_size=self._filter_size,
                inputs_channel=self._inputs_channel, shape=self._shape,
                c_h_channel=self._c_h_channel, forget_bias=self._forget_bias,
                num_steps=self._num_steps, batch_size=self._batch_size)
        self._resnet_lr = ResNet50_lr(include_top=False)
        self._resnet_hr = ResNet50_hr(include_top=False)

    def _init_holder(self):
        self._labels_holder = tf.placeholder(name='labels', 
                shape=(None, self._num_steps, self._shape[0]*8, self._shape[1]*8, 1),
                dtype=tf.float32)

    def _decode_predicts(self, predicts):
        scanpath = []
        predicts = np.array(predicts)
        for i in range(self._batch_size):
            scanpath_img = []
            for j in range(self._num_steps):
                coord = np.argmax(predicts[i, j, :, :, :])
                y = coord // 512
                x = coord % 512
                scanpath_img.append([x, y])
            scanpath.append(scanpath_img)
        scanpath = np.array(scanpath)
        return scanpath

    def _compute_loss(self):
        preds_pack = self._preds(mode='train')
        preds = preds_pack[0]
        labels = self._labels_holder
        weight = labels >= 0.0
        weight = tf.cast(weight, dtype=tf.float32)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=preds)
        loss = weight * loss
        num = tf.reduce_max(weight, axis=[2, 3, 4])
        num = (tf.reduce_sum(num)+1)
        loss = tf.reduce_sum(loss)
        loss = loss / self._batch_size / num
        #loss_construct = tf.losses.mean_squared_error(preds_pack[1], preds_pack[2])
        #loss = 0.1 * loss + 0.9 * loss_construct
        return loss

    def _generate_feed_dict(self, idxs):
        features_lr = []
        features_hr = []
        scanpaths = np.load(os.path.join(self._feature_dir[2]))
        for idx in idxs:
            feature = np.load(os.path.join(self._feature_dir[0], str(idx[0])+'.npy'))
            feature = preprocess_input(feature[np.newaxis, :, :, :], backend=K)
            feature = self._resnet_lr.predict(feature)[0, :, :, :]
            features_lr.append(feature)
            feature = np.load(os.path.join(self._feature_dir[1], str(idx[0])+'.npy'))
            features_t = []
            for i in range(self._num_steps):
                coord = scanpaths[idx[0]][idx[1]][i][:]
                coord_x = coord[0]
                coord_y = coord[1]
                if coord_x >= 512:
                    coord_x = 511
                if coord_y >= 384:
                    coord_y = 383
                coord = [coord_x, coord_y]
                padding_feature = np.zeros((384+self._shape[2]-1, 512+self._shape[3]-1, 3))
                padding_feature[(self._shape[2]-1)//2: -(self._shape[2]-1)//2,
                    (self._shape[3]-1)//2: -(self._shape[3]-1)//2, :] = feature[:, :, :]
                f = padding_feature[int(coord[1]): int(coord[1])+self._shape[2],
                        int(coord[0]): int(coord[0])+self._shape[3], :]
                f = preprocess_input(f[np.newaxis, :, :, :], backend=K)
                f = self._resnet_hr.predict(f)[0, :, :, :]
                features_t.append(f)
            features_hr.append(features_t)
        features_lr = np.array(features_lr)
        features_hr = np.array(features_hr)

        scanpaths = []
        in_scanpaths = []
        for idx in idxs:
            s = np.load(os.path.join(self._feature_dir[3], str(idx[0])+'.npy'))
            in_scanpaths.append(s[idx[1]][0: self._num_steps, :, :, np.newaxis])
            scanpaths.append(s[idx[1]][1: self._num_steps+1, :, :, np.newaxis])
        scanpaths = np.array(scanpaths)
        in_scanpaths = np.array(in_scanpaths)
        lr_c_init = np.zeros((np.shape(idxs)[0], self._shape[0], self._shape[1], self._c_h_channel[0]))
        lr_h_init = np.zeros((np.shape(idxs)[0], self._shape[0], self._shape[1], self._c_h_channel[1]))
        hr_c_init = np.zeros((np.shape(idxs)[0], self._shape[2], self._shape[3], self._c_h_channel[2]))
        hr_h_init = np.zeros((np.shape(idxs)[0], self._shape[2], self._shape[3], self._c_h_channel[3]))
        in_c_init = np.zeros((np.shape(idxs)[0], 384, 512, self._c_h_channel[3]))
        in_h_init = np.zeros((np.shape(idxs)[0], 384, 512, self._c_h_channel[3]))
        feed_dict = {self._preds.lr_h_init: lr_h_init, self._preds.hr_h_init: hr_h_init,
                self._preds.lr_c_init: lr_c_init, self._preds.hr_c_init: hr_c_init,
                self._preds.in_c_init: in_c_init, self._preds.in_h_init: in_h_init,
                self._preds.lr_inputs: features_lr, self._preds.hr_inputs: features_hr,
                self._labels_holder: scanpaths, self._preds.in_inputs: in_scanpaths}
        return feed_dict

