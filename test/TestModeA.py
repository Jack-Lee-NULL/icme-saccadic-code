#
#
# file: TestModeA.py
# author: Jingquan Lee
# date: 2019-03-10
#
#

from model import *

class TestModeA:

    def __init__(self, trained_model, feature_dir, 
            shape, filter_size, inputs_channel, c_h_channel,
            forget_bias, init_hidden, num_steps, idxs, batch_size,
            preds_path):
        self._trained_model = trained_model
        self._feature_dir = feature_dir
        self._shape = shape
        self._init_hidden = init_hidden
        self._num_steps = num_steps
        self._idxs = idxs
        self._batch_size = batch_size
        self._preds_path = preds_path
        self._predictor = BasicSaccadicModel.BasicSaccadicModel(
                shape=shape, filter_size=filter_size, inputs_channel=inputs_channel,
                c_h_channel=c_h_channel, forget_bias=forget_bias, num_steps=num_steps)

    def predicts(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        n_iters = np.shape(self._idxs)[0] // self._batch_size
        predictor = self._predictor()
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self._trained_model)
            preds = []
            for i in range(n_iters):
                idxs = self._idxs[i * self._batch_size, (i + 1) * self._batch_size, :]
                feed_dict = self._generate_feed_dict(idxs)
                pred = sess.run(predictor, feed_dict)
                preds.append(pred)
            if np.shape(self._idxs)[0] % self._batch_size != 0:
                idxs = self._idxs[i * self._batch_size, np.shape(self._idxs)[0], :]
                feed_dict = self._generate_feed_dict(idxs)
                pred = sess.run(predictor, feed_dict)
                preds.append(pred)               
        preds = np.concatenate(preds, axis=0)
        np.save(self._preds_path, preds)
        print ("Predictions have been saved to", self._preds_path)
        
    def _generate_feed_dict(self, idxs):
        c_init = np.zeros((self._batch_size, self._shape[0], self._shape[1], self._c_h_channel))
        h_init = self._init_hidden[idxs[:, 0], :, :, np.newaxis]
        features = []
        for idx in idxs:
            feature = np.load(os.path.join(self._feature_dir, str(idx[0])+'.npy'))
            features.append(feature)
        features = np.array(features)
        feed_dict = {self._predictor.c_init: c_init, self._predictor.h_init: h_init,
                self._predictor.inputs: features}
        return feed_dict

    def decode_preds(self):
        predicts[:, :, 0] = predicts[:, :, 0] * self._shape[1]
        predicts[:, :, 1] = predicts[:, :, 1] * self._shape[0]
        predicts = predicts.astype('int32')
        predicts = np.concatenate([predicts[:, :, 0], predicts[:, :, 1]], axis=1)
        return predicts

