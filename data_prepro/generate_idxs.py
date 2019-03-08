#
#
# file: generate_idxs.py
# author: Jingquan Lee
# date: 2019-03-08
#
#

import os

import numpy as np

from __init__ import *

def generate_id(scanpath):
    idxs = []
    for i in range(len(scanpath)):
        for j in range(len(scanpath[i])):
            idxs.append([i, j])
    idxs = np.array(idxs)
    return idxs

if __name__ == '__main__':
    SCANPATH_PATH = 'TD_384_512_scanpath.npy'
    OUTPUT_PATH = 'TD_idxs.npy'

    SCANPATH_PATH = os.path.join(DATA_DIR, SCANPATH_PATH)
    OUTPUT_PATH = os.path.join(DATA_DIR, OUTPUT_PATH)
    scanpath = np.load(SCANPATH_PATH)
    idxs = generate_id(scanpath)
    print(np.shape(idxs))
    np.save(OUTPUT_PATH, idxs)
