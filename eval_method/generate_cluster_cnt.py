#
#
# file: generate_cluster_cnt.py
# author: Jingquan Lee
# date: 2019-03-12
#
#

import os

import numpy as np
from scipy import io
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn import metrics

from __init__ import *

def generate_cluster_cnt(gt_scanpath, bw):
    """generate the cluster centers of each image based on 
    ground truth scanpath
    Args:
        -gt_scanpath: a list, (imgs, person, step, coordination),
        groundtruth human scanpath
    Returns:
        -cluster_cnt: a 3-dims Array, (img, x, y), cluster centers
        of all imgs
        -bandwidth: a 1-dim Array, (id of img), local area radius of mean-shift
    """
    cluster_cnt = []
    bandwidth = []
    for img_fix in gt_scanpath:
        img_fix = np.concatenate(img_fix, axis=0)
        img_fix = img_fix[:, 0: 2]
        #bw = estimate_bandwidth(img_fix, quantile=0.05, n_samples=100)
        cnt_bw = []
        scores = []
        for b in bw:
            ms = MeanShift(bandwidth=b, bin_seeding=True)
            ms.fit(img_fix)
            cnt_bw.append(ms.cluster_centers_)
            scores.append(metrics.calinski_harabaz_score(img_fix, ms.labels_))
        max_index = np.argmax(scores)
        cluster_cnt.append(cnt_bw[max_index])
        bandwidth.append(bw[max_index])
    cluster_cnt = np.array(cluster_cnt)
    bandwidth = np.array(bandwidth)
    return cluster_cnt, bandwidth

if __name__ == '__main__':
    GT_SCANPATH_PATH = os.path.join(DATA_DIR, 'ASD_768_1024_scanpath.npy')
    OUTPUT_SCANPATH_CNT_PATH = os.path.join(DATA_DIR, 'ASD_gt_test_768_1024_ms_cnt_for_eval.mat')
    
    gt_scanpath = np.load(GT_SCANPATH_PATH)
    bws = np.array(range(50, 105, 5))
    cluster_cnt, bandwidth = generate_cluster_cnt(gt_scanpath[0: 10], bws)
    print(np.shape(cluster_cnt))
    print(np.shape(bandwidth))
    print(cluster_cnt[0])
    print(bandwidth)

    io.savemat(OUTPUT_SCANPATH_CNT_PATH, {'cluster_cnt': cluster_cnt, 'bandwidth': bandwidth})
    

