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
from test import *

class Main:

    def __init__(self, section, mode):
        self._section = section
        self._mode = mode
        self._config = ConfigParser()
        self._config.read('config.ini')
        self._train_modes = {'A': self._train_A, 'B': self._train_B,
                'C': self._train_C, 'D': self._train_D}
        self._test_modes = {'A': self._test_A, 'B': self._test_B,
                'D': self._test_D}
        self._train_mode = self._config.get(self._section, 'train_mode', fallback='A')
        self._test_mode = self._config.get(self._section, 'test_mode', fallback='A')

        self.learning_rate = self._config.getfloat(self._section, 'learning_rate', fallback=0.0005)
        self.epochs = self._config.getint(self._section, 'epochs', fallback=20)
        self.batch_size = self._config.getint(self._section, 'batch_size', fallback=10)
        shape_r = self._config.getint(self._section, 'shape_r', fallback=384)
        shape_c = self._config.getint(self._section, 'shape_c', fallback=512)
        self.shape = (shape_r, shape_c)
        self.print_every = self._config.getint(self._section, 'print_every', fallback=1)
        self.save_every = self._config.getint(self._section, 'save_every', fallback=1)
        self.log_path = self._config.get(self._section, 'log_path')
        filter_size_r = self._config.get(self._section, 'filter_size_r', fallback=3)
        filter_size_c = self._config.get(self._section, 'filter_size_c', fallback=3)
        self.filter_size = (filter_size_r, filter_size_c)
        self.inputs_channel = self._config.getint(self._section, 'inputs_channel', fallback=64)
        self.c_h_channel = self._config.getint(self._section, 'c_h_channel', fallback=1)
        self.forget_bias = self._config.getfloat(self._section, 'forget_bias', fallback=1.0)
        self.init_hidden_path = self._config.get(self._section, 'init_hidden_path', fallback=None)
        self.save_model_path = self._config.get(self._section, 'save_model_path')
        self.pretrained_model = self._config.get(self._section, 'pretrained_model', fallback=None)
        self.feature_dir = self._config.get(self._section, 'feature_dir', fallback=None)
        self.scanpath_path = self._config.get(self._section, 'scanpath_path', fallback=None)
        self.idxs_path = self._config.get(self._section, 'idxs_path')
        self.num_steps = self._config.getint(self._section, 'num_steps', fallback=8)
        self.num_validation = self._config.getint(self._section, 'num_validation', fallback=10)

        self.trained_model = self._config.get(self._section, 'trained_model', fallback=None)
        self.test_feature_dir = self._config.get(self._section, 'test_feature_dir', fallback=None)
        self.test_init_hidden = self._config.get(self._section, 'test_init_hidden', fallback=None)
        self.test_idxs = self._config.get(self._section, 'test_idxs', fallback=None)
        self.preds_path = self._config.get(self._section, 'preds_path', fallback=None)

    def _read_B_data(self):
        shape_lr_r = self._config.getint(self._section, 'shape_lr_r', fallback=48)
        shape_lr_c = self._config.getint(self._section, 'shape_lr_c', fallback=64)
        shape_hr_r = self._config.getint(self._section, 'shape_hr_r', fallback=16)
        shape_hr_c = self._config.getint(self._section, 'shape_hr_c', fallback=16)
        self.shape = (shape_lr_r, shape_lr_c, shape_hr_r, shape_hr_c)
        filter_size_lr_r = self._config.getint(self._section, 'filter_size_lr_r', fallback=16)
        filter_size_lr_c = self._config.getint(self._section, 'filter_size_lr_c', fallback=16)
        filter_size_hr_r = self._config.getint(self._section, 'filter_size_hr_r', fallback=16)
        filter_size_hr_c = self._config.getint(self._section, 'filter_size_hr_c', fallback=16)
        self.filter_size = (filter_size_lr_r, filter_size_lr_c, filter_size_hr_r,
                filter_size_hr_c)
        inputs_channel_lr = self._config.getint(self._section, 'inputs_channel_lr', fallback=2048)
        inputs_channel_hr = self._config.getint(self._section, 'inputs_channel_hr', fallback=64)
        self.inputs_channel = (inputs_channel_lr, inputs_channel_hr)
        c_channel_lr = self._config.getint(self._section, 'c_channel_lr', fallback=1)
        c_channel_hr = self._config.getint(self._section, 'c_channel_hr', fallback=1)
        h_channel_lr = self._config.getint(self._section, 'h_channel_lr', fallback=1)
        h_channel_hr = self._config.getint(self._section, 'h_channel_hr', fallback=1)
        self.c_h_channel = (c_channel_lr, h_channel_lr, c_channel_hr, h_channel_hr)
        forget_bias_lr = self._config.getfloat(self._section, 'forget_bias_lr', fallback=1.0)
        forget_bias_hr = self._config.getfloat(self._section, 'forget_bias_hr', fallback=1.0)
        self.forget_bias = (forget_bias_lr, forget_bias_hr)
        scanpath_path_lr = self._config.get(self._section, 'scanpath_path_lr')
        scanpath_path_hr = self._config.get(self._section, 'scanpath_path_hr')
        self.scanpath = (np.load(scanpath_path_lr), np.load(scanpath_path_hr))
        feature_dir_lr = self._config.get(self._section, 'feature_dir_lr')
        feature_dir_hr = self._config.get(self._section, 'feature_dir_hr')
        self.feature_dir = (feature_dir_lr, feature_dir_hr)
        self.idxs = np.load(self.idxs_path)

    def _test_A(self):
        init_hidden = np.load(self.test_init_hidden)
        idxs = np.load(self.test_idxs)
        predictor = TestModeA.TestModeA(trained_model=self.trained_model,
                feature_dir=self.feature_dir, shape=self.shape,
                filter_size=self.filter_size, inputs_channel=self.inputs_channel,
                c_h_channel=self.c_h_channel, forget_bias=self.forget_bias,
                init_hidden=init_hidden, num_steps=self.num_steps,
                idxs=idxs, batch_size=self.batch_size, preds_path=self.preds_path)
        predictor.predicts()

    def _test_B(self):
        self._read_B_data()
        idxs = np.load(self.test_idxs)
        predictor = TestModeB.TestModeB(trained_model=self.trained_model,
                feature_dir=self.feature_dir, shape=self.shape,
                filter_size=self.filter_size, inputs_channel=self.inputs_channel,
                c_h_channel=self.c_h_channel, forget_bias=self.forget_bias,
                num_steps=self.num_steps,
                idxs=idxs, batch_size=self.batch_size, preds_path=self.preds_path)       
        predictor.predicts()

    def _test_D(self):
        self._read_B_data()
        idxs = np.load(self.test_idxs)
        predictor = TestModeD.TestModeD(trained_model=self.trained_model,
                feature_dir=self.feature_dir, shape=self.shape,
                filter_size=self.filter_size, inputs_channel=self.inputs_channel,
                c_h_channel=self.c_h_channel, forget_bias=self.forget_bias,
                num_steps=self.num_steps,
                idxs=idxs, batch_size=self.batch_size, preds_path=self.preds_path)       
        predictor.predicts()

    def _train_A(self):
        if self.init_hidden_path != None:
            init_hidden = np.load(self.init_hidden_path)
        else:
            init_hidden = None
        scanpath = np.load(self.scanpath_path)
        idxs = np.load(self.idxs_path)
        if not os.path.exists(self.save_model_path):
            os.mkdir(self.save_model_path)
        
        train = TrainModeA.TrainModeA(learning_rate=self.learning_rate, epochs=self.epochs,
                batch_size=self.batch_size, shape=self.shape, print_every=self.print_every, 
                save_every=self.save_every, log_path=self.log_path,
                filter_size=self.filter_size, inputs_channel=self.inputs_channel, 
                c_h_channel=self.c_h_channel, forget_bias=self.forget_bias, 
                init_hidden=init_hidden, save_model_path=self.save_model_path,
                pretrained_model=self.pretrained_model, feature_dir=self.feature_dir,
                scanpath=scanpath, idxs=idxs, num_steps=self.num_steps, 
                num_validation=self.num_validation)
        train.train()

    def _train_C(self):
        if self.init_hidden_path != None:
            init_hidden = np.load(self.init_hidden_path)
        else:
            init_hidden = None
        scanpath = np.load(self.scanpath_path)
        idxs = np.load(self.idxs_path)
        if not os.path.exists(self.save_model_path):
            os.mkdir(self.save_model_path)
        
        train = TrainModeC.TrainModeC(learning_rate=self.learning_rate, epochs=self.epochs,
                batch_size=self.batch_size, shape=self.shape, print_every=self.print_every, 
                save_every=self.save_every, log_path=self.log_path,
                filter_size=self.filter_size, inputs_channel=self.inputs_channel, 
                c_h_channel=self.c_h_channel, forget_bias=self.forget_bias, 
                init_hidden=init_hidden, save_model_path=self.save_model_path,
                pretrained_model=self.pretrained_model, feature_dir=self.feature_dir,
                scanpath=scanpath, idxs=idxs, num_steps=self.num_steps, 
                num_validation=self.num_validation)
        train.train()

    def _train_B(self):
        self._read_B_data()
        
        train = TrainModeB.TrainModeB(learning_rate=self.learning_rate, epochs=self.epochs,
                batch_size=self.batch_size, shape=self.shape, print_every=self.print_every,
                save_every=self.save_every, log_path=self.log_path, filter_size=self.filter_size,
                inputs_channel=self.inputs_channel, c_h_channel=self.c_h_channel,
                forget_bias=self.forget_bias, save_model_path=self.save_model_path,
                pretrained_model=self.pretrained_model, feature_dir=self.feature_dir,
                scanpath=self.scanpath, idxs=self.idxs, num_steps=self.num_steps, 
                num_validation=self.num_validation)
        train.train()

    def _train_D(self):
        self._read_B_data()
        train = TrainModeD.TrainModeD(learning_rate=self.learning_rate, epochs=self.epochs,
                batch_size=self.batch_size, shape=self.shape, print_every=self.print_every,
                save_every=self.save_every, log_path=self.log_path, filter_size=self.filter_size,
                inputs_channel=self.inputs_channel, c_h_channel=self.c_h_channel,
                forget_bias=self.forget_bias, save_model_path=self.save_model_path,
                pretrained_model=self.pretrained_model, feature_dir=self.feature_dir,
                scanpath=self.scanpath, idxs=self.idxs, num_steps=self.num_steps, 
                num_validation=self.num_validation)
        train.train()

    def run(self):
        if self._mode == 'train':
            self._train_modes[self._train_mode]()
        else:
            self._test_modes[self._test_mode]()

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    if len(sys.argv) != 3:
        raise TypeError('required 2 parameters,', len(sys.argv), 'given')
    if sys.argv[2] != 'train' and sys.argv[2] != 'test':
        raise TypeError('2nd parameter required \'test\' or \'train\'')
    main = Main(sys.argv[1], sys.argv[2])
    main.run()
