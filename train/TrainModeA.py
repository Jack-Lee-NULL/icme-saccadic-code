#
#
# file: TrainModeA.py
# author: Jingquan Lee
# date: 2019-03-08
#
#

import time
import os

import numpy as np
import tensorflow as tf

from model import *

class TrainModeA:

    def __init__(self, learning_rate=0.0005, epochs=20, batch_size=10, shape=(384, 512),
                 print_every=1, save_every=1, log_path=None, filter_size=(3, 3),
                 inputs_channel=64, c_h_channel=1, forget_bias=1.0, init_hidden=None,
                 save_model_path=None, pretrained_model=None, feature_dir=None,
                 scanpath=None, idxs=None, num_steps=8, num_validation=None,
                 output_path=None):
        self._learning_rate = learning_rate
        self._epochs = epochs
        self._batch_size = batch_size
        self._print_every = print_every
        self._save_every = save_every
        self._log_path = log_path
        self._feature_dir = feature_dir
        self._scanpath = scanpath
        self._idxs = idxs
        self._num_validation = num_validation
        self._init_hidden = init_hidden
        self._save_model_path = save_model_path
        self._pretrained_model = pretrained_model
        self._c_h_channel = c_h_channel
        self._num_steps = num_steps
        self._shape = shape
        self._filter_size = filter_size
        self._inputs_channel = inputs_channel
        self._forget_bias = forget_bias
        self._output_path = output_path

        self._init_model()
        self._init_holder()       

    def _init_model(self):
        self._preds = BasicSaccadicModel.BasicSaccadicModel(
                shape=self._shape, filter_size=self._filter_size, 
                inputs_channel=self._inputs_channel,
                c_h_channel=self._c_h_channel, forget_bias=self._forget_bias,
                num_steps=self._num_steps)

    def _init_holder(self):
        self._labels_holder=tf.placeholder(name='labels', shape=(None, self._num_steps, 2),
                dtype=tf.float32)

    def train(self):
        predicts = self._preds(mode='test')
        loss = self._compute_loss()
        optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate)
        train_op = optimizer.minimize(loss)

        batch_loss = tf.summary.scalar('batch_loss', loss)
        validation_loss = tf.summary.scalar('validation_loss', loss)

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        prev_loss = -1
        current_loss = 0.0
        start_t = time.time()
        with tf.Session(config=config) as sess:
            tf.initializers.global_variables().run()
            summary_writer = tf.summary.FileWriter(self._log_path, graph=tf.get_default_graph())
            saver = tf.train.Saver(max_to_keep=20)
            if self._pretrained_model != None:
                print('loading pretrained model:', self._pretrained_model)
                saver.restore(sess, self._pretrained_model)
            print('Learning rate:', self._learning_rate)
            print('Batch size:', self._batch_size)
            print('The number of epochs:', self._epochs)
            print('The number of validating images :', self._num_validation)
            print('Log path:', self._log_path)
            print('Print every', self._print_every, 'epochs')
            print('Save model to', self._save_model_path, 'every',
                self._save_every, 'epochs')
            num_validation_idxs = 0
            for i in range(self._epochs):
                pre_num_validation = num_validation_idxs
                num_validation_idxs = np.shape(np.argwhere(self._idxs[:, 0] < (i+1)*self._num_validation))[0]
                validation_idxs = self._idxs[pre_num_validation: num_validation_idxs, :]
                train_idxs = [self._idxs[num_validation_idxs: np.shape(self._idxs)[0], :],
                        self._idxs[0: pre_num_validation, :]]
                train_idxs = np.concatenate(train_idxs, axis=0)
                n_iter_per_epochs = np.shape(train_idxs)[0] // self._batch_size
                np.random.shuffle(train_idxs)
                for j in range(n_iter_per_epochs):
                    idxs = train_idxs[j * self._batch_size: (j + 1) * self._batch_size, :]
                    feed_dict = self._generate_feed_dict(idxs)
                    _, l = sess.run([train_op, loss], feed_dict)
                    current_loss += l
                    if j % 2 == 0:
                        summary = sess.run(batch_loss, feed_dict)
                        summary_writer.add_summary(summary, i * n_iter_per_epochs + j)
                    if j % 4 == 0:
                        np.random.shuffle(validation_idxs)
                        idxs = validation_idxs[0: self._batch_size, :]
                        feed_dict = self._generate_feed_dict(idxs)
                        summary = sess.run(validation_loss, feed_dict)
                        summary_writer.add_summary(summary, i * n_iter_per_epochs + j)
                if i % self._save_every == 0:
                    saver.save(sess, os.path.join(self._save_model_path, 'model'), global_step=i)
                    print('Model has been saved to', self._save_model_path)
                if i % self._print_every == 0:
                    print("Previous loss:", prev_loss)
                    print("Current loss:", current_loss)
                    print("Elapsed time:", time.time() - start_t)
                    prev_loss = current_loss
                    current_loss = 0.0
                    predictions = []
                    print('generate validation result')
                    for k in range(self._batch_size):
                        idxs = validation_idxs[k: k+1, :]
                        feed_dict = self._generate_feed_dict(idxs)
                        preds = sess.run(predicts, feed_dict)
                        if self._output_path != None:
                            np.save(self._output_path, preds[2])
                            np.save('./c_img.npy', preds[1])
                            np.save('./preds_img.npy', preds[0])
                        preds = self._decode_predicts(preds[0])
                        predictions.append(preds)
                    predictions = np.concatenate(predictions, axis=0)
                    print('generate labels')
                    labels = feed_dict[self._labels_holder]
                    labels = self._decode_predicts(labels)
                    print(predictions[-self._batch_size: np.shape(predictions)[0], :])
                    print(labels)
                    start_t = time.time()
                
    def _decode_predicts(self, predicts):
        predicts[:, :, 0] = predicts[:, :, 0] * self._shape[1]
        predicts[:, :, 1] = predicts[:, :, 1] * self._shape[0]
        predicts = predicts.astype('int32')
        predicts = np.concatenate([predicts[:, :, 0], predicts[:, :, 1]], axis=1)
        return predicts
        
    def _compute_loss(self):
        preds = self._preds(mode='train') 
        labels = self._labels_holder
        loss = 0.0
        weight = labels > 0
        weight = tf.cast(weight, dtype=tf.float32)
        preds = tf.multiply(preds, weight)
        loss = tf.losses.mean_squared_error(labels, preds)
        loss = loss * self._num_steps
        return loss

    def _generate_feed_dict(self, idxs):
        features = []
        for idx in idxs:
            feature = np.load(os.path.join(self._feature_dir, str(idx[0])+'.npy'))
            features.append(feature)
        features = np.array(features)
        scanpaths = []
        for idx in idxs:
            scanpath = self._scanpath[idx[0]][idx[1]][:, :]
            scanpaths.append(scanpath)
        scanpaths = np.array(scanpaths)
        scanpaths = scanpaths[:, 0: self._num_steps, 0: 2]
        h_init = self._init_hidden[idxs[:, 0], :, :, np.newaxis]
        c_init = np.zeros((np.shape(idxs)[0], self._shape[0], self._shape[1], self._c_h_channel))
        feed_dict = {self._preds.c_init: c_init, self._preds.h_init: h_init,
                self._preds._inputs: features, self._labels_holder: scanpaths}
        return feed_dict
