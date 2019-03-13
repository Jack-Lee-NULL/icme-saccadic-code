#
#
# file: TrainModeB.py
# author: Jingquan Lee
# date: 2019-03-13
#
#

from TrainModeA import TrainModeA

class TrainModeB(TrainModeA):

    def __init__(self, learning_rate=0.0005, epochs=20, batch_size=10, shape=(48, 64, 16, 16),
                 print_every=1, save_every=1, log_path=None, filter_size=(3, 3, 3, 3),
                 inputs_channel=(64, 64), c_h_channel=1, forget_bias=1.0, init_hidden=None,
                 save_model_path=None, pretrained_model=None, feature_dir=None,
                 scanpath=None, idxs=None, num_steps=8, num_validation=None):
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
        pass
