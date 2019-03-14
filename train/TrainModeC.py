#
#
# file: TrainModeC.py
# date: 2019-03-14
# author: Jingquan Lee
#
#

from TrainModeA import TrainModeA

class TrainModeC(TrainModeA):
    """Train BasicSaccadicModel with human first fixation as start point
    """

    def _generate_feed_dict(self, idxs):
        features = []
        for idx in idxs:
            feature = np.load(os.path.join(self._feature_dir, str(idx[0])+'.npy'))
            features.append(feature)
        features = np.array(features)
        scanpaths = []
        for idx in idxs:
            scanpath = self._scanpath[idx[0]][idx[1]][1: 9, :]
            scanpaths.append(scanpath)
        scanpaths = np.array(scanpaths)
        scanpaths = scanpaths[:, 0: self._num_steps, 0: 2]        
        #h_init = self._init_hidden[idxs[:, 0], :, :, np.newaxis]
        h_init = []
        for idx in idxs:
            init = np.zeros(self._shape)
            coord = [self._scanpath[idx[0]][idx[1]][0, 0] * self._shape[1],
                    self._scanpath[idx[0]][idx[1]][0, 1] * self._shape[0]]
            coord = np.array(coord) - 1
            init[int(coord[1])][int(coord[0])] = 1.0
            kernel = cv2.getGaussianKernel(15, sigma = -1)
            init = cv2.filter2D(init, ddepth = -1, kernel = kernel)
            init = cv2.filter2D(init, ddepth = -1, kernel = kernel.T)
            h_init.append(init[:, :, np.newaxis])
        h_init = np.array(h_init)
        c_init = np.zeros((np.shape(idxs)[0], self._shape[0], self._shape[1], self._c_h_channel))
        feed_dict = {self._preds.c_init: c_init, self._preds.h_init: h_init,
                self._preds._inputs: features, self._labels_holder: scanpaths}
        return feed_dict
