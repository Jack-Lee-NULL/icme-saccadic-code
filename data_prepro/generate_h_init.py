#
#
# file: generate_h_init.py
# author: Jingquan Lee
# date: 2019-03-08
#
#

import os

import cv2
import numpy as np

from __init__ import *

def generate_h_init(init_coord, img_shape = (384, 512),
                    kernel_size = 15, sigma = -1):
    init_h = np.zeros(img_shape)
    init_h[init_coord[1]][init_coord[0]] = 1.0
    kernel = cv2.getGaussianKernel(kernel_size, sigma = sigma)
    init_h = cv2.filter2D(init_h, ddepth = -1, kernel = kernel)
    return init_h

if __name__ == '__main__':
    OUTPUT_INIT_H = './init_h_384_512_-1.npy'
    OUTPUT_INIT_H = os.path.join(DATA_DIR, OUTPUT_INIT_H)

    init_h = []
    for i in range(300):
        init_h.append(generate_h_init((256, 192)))

    init_h = np.array(init_h)
    np.save(OUTPUT_INIT_H, init_h)
