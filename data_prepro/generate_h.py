#
#
# file: generate_h.py
# author: Jingquan Lee
# date: 2019-03-26
#
#

import os

import cv2
import numpy as np

def generate_h(scanpath, shape):
    h = []
    scanpath = scanpath.astype('int32')
    img_o = np.zeros(shape)
    kernel = cv2.getGaussianKernel(199, -1)
    for s in scanpath:
        if s[1] != 0 or s[0] !=0:
            img = np.zeros(shape)
            img[s[1]-1, s[0]-1] = 1.0
            img = cv2.filter2D(img, -1, kernel)
            img = cv2.filter2D(img, -1, kernel.T)
            img = img / np.max(img)
            #img_o = 0.5 * img_o
            #img_o = np.amax([img, img_o], axis=0)
            img_o = img
        else:
            img_o = -np.ones(np.shape(img_o))
            #img_o = 0.5 * img_o
        h.append(img_o)
    return h

if __name__ == '__main__':

    scanpath = '/data/jqLee/data/ASD_384_512_scanpath_merged_fixations.npy'
    output_dir = '/data/jqLee/data/ASD_384_512_scanpath_img/'
    img_dir = '/home/camerart/jqLee/exchange/h_img'
    shape = (384, 512)

    scanpath = np.load(scanpath)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    i = 0
    for scanpath_img in scanpath:
        h_per = []
        for scanpath_per in scanpath_img:
            #scanpath_per[:, 0] = scanpath_per[:, 0] * shape[1]
            #scanpath_per[:, 1] = scanpath_per[:, 1] * shape[0]
            img = generate_h(scanpath_per, shape)
            h_per.append(img)
        print(np.shape(h_per))
        np.save(os.path.join(output_dir, str(i)+'.npy'), h_per)
        print ('Has been saved to', os.path.join(output_dir, str(i)+'.npy'))
        i += 1
 
    i = 0
    for s in h_per[0]:
        cv2.imwrite(os.path.join(img_dir, str(i)+'.png'), (s*255).astype('uint8'))
        i += 1
