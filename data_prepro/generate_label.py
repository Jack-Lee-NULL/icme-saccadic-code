#
#
# file: generate_label.py
# author: Jingquan Lee
# date: 2019-04-18
#
#

import os

import cv2
import numpy as np

if __name__ == '__main__':
    scanpath_img = '/data/jqLee/data/ASD_384_512_scanpath_img'
    saliency_map = '/data/jqLee/data/ASD_saliency_map.npy'
    output_dir = '/data/jqLee/data/ASD_384_512_scanpath_img_label'
    num = 300
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    s = np.load(saliency_map)

    for i in range(num):
        print(i)
        scanpath = np.load(os.path.join(scanpath_img, str(i)+'.npy'))
        scanpath_per = []
        m = s[i, :, :]
        m = m / np.max(m)
        for j in range(np.shape(scanpath)[0]):
            scanpath_t = []
            for k in range(np.shape(scanpath)[1]):
                if np.max(scanpath[j, k, :, :]) >= 0:
                    c = np.argmax(scanpath[j, k, :, :])
                    img = np.array(m)
                    img = m * scanpath[j, k, :, :]
                    img = (img - np.min(img)) / (np.max(img) - np.min(img))
                else:
                    img = scanpath[j, k, :, :]
                scanpath_t.append(img)
            scanpath_per.append(scanpath_t)
        np.save(os.path.join(output_dir, str(i)+'.npy'), scanpath_per)
