#
#
# file: main.py
# author: Jingquan Lee
# date: 2019-03-09
#
# entrance program of training, testing model
#

import os
import sys
from configparser import ConfigParser

import numpy as np

from train import *

class Main:

    def __init__(self, section, mode):
        self._section = section
        self._mode = mode
        self._config = ConfigParser()
        self._config.read('config.ini')
        self._train_mode = {}
        self._test_mode = {}

    def _test(self):
        pass

    def _train(self):
        learning_rate = self._config.getfloat(self._section, 'learning_rate', fallback=0.0005)
        epochs = self._config.getint(self._section, 'epochs', fallback=20)
        batch_size = self._config.getint(self._section, 'batch_size', fallback=10)
        shape_r = self._config.getint(self._section, 'shape_r', fallback=384)
        shape_c = self._config.getint(self._section, 'shape_c', fallback=512)
        shape = (shape_r, shape_c)
        print_every = self._config.getint(self._section, 'print_every', fallback=1)
        save_every = self._config.getint(self._section, 'save_every', fallback=1)
        log_path = self._config.get(self._section, 'log_path')
        filter_size_r = self._config.get(self._section, 'filter_size_r', fallback=3)
        filter_size_c = self._config.get(self._section, 'filter_size_c', fallback=3)
        filter_size = (filter_size_r, filter_size_c)
        inputs_channel = self._config.getint(self._section, 'inputs_channel', fallback=64)
        c_h_channel = self._config.getint(self._section, 'c_h_channel', fallback=1)
        forget_bias = self._config.getfloat(self._section, 'forget_bias', fallback=1.0)
        init_hidden_path = self._config.get(self._section, 'init_hidden_path')
        save_model_path = self._config.get(self._section, 'save_model_path')
        pretrained_model = self._config.get(self._section, 'pretrained_model', fallback=None)
        feature_dir = self._config.get(self._section, 'feature_dir')
        scanpath_path = self._config.get(self._section, 'scanpath_path')
        idxs_path = self._config.get(self._section, 'idxs_path')
        num_steps = self._config.getint(self._section, 'num_steps', fallback=8)
        num_validation = self._config.getint(self._section, 'num_validation', fallback=10)

        init_hidden = np.load(init_hidden_path)
        scanpath = np.load(scanpath_path)
        idxs = np.load(idxs_path)
        if not os.path.exists(save_model_path):
            os.mkdir(save_model_path)
        
        train = TrainModeA.TrainModeA(learning_rate=learning_rate, epochs=epochs, batch_size=batch_size,
                shape=shape, print_every=print_every, save_every=save_every, log_path=log_path,
                filter_size=filter_size, inputs_channel=inputs_channel, c_h_channel=c_h_channel,
                forget_bias=forget_bias, init_hidden=init_hidden, save_model_path=save_model_path,
                pretrained_model=pretrained_model, feature_dir=feature_dir,
                scanpath=scanpath, idxs=idxs, num_steps=num_steps, num_validation=num_validation)
        train.train()


    def run(self):
        if self._mode == 'train':
            self._train()
        else:
            self._test()

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
    if len(sys.argv) != 3:
        raise TypeError('required 2 parameters,', len(sys.argv), 'given')
    if sys.argv[2] != 'train' and sys.argv != 'test':
        raise TypeError('2nd parameter required \'test\' or \'train\'')
    main = Main(sys.argv[1], sys.argv[2])
    main.run()
