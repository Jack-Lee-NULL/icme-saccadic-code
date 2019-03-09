from __future__ import division
import pickle
import os
import cv2
import numpy as np
import scipy.io
import scipy.ndimage

from __init__ import *

def padding(img, shape_r=224, shape_c=224, channels=3):
    img_padded = np.zeros((shape_r, shape_c, channels), dtype=np.uint8)
    if channels == 1:
        img_padded = np.zeros((shape_r, shape_c), dtype=np.uint8)

    original_shape = img.shape
    rows_rate = original_shape[0]/shape_r
    cols_rate = original_shape[1]/shape_c

    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * shape_r) // original_shape[0]
        img = cv2.resize(img, (new_cols, shape_r))
        if new_cols > shape_c:
            new_cols = shape_c
        img_padded[:, ((img_padded.shape[1] - new_cols) // 2):((img_padded.shape[1] - new_cols) // 2 + new_cols)] = img
    else:
        new_rows = (original_shape[0] * shape_c) // original_shape[1]
        img = cv2.resize(img, (shape_c, new_rows))
        if new_rows > shape_r:
            new_rows = shape_r
        img_padded[((img_padded.shape[0] - new_rows) // 2):((img_padded.shape[0] - new_rows) // 2 + new_rows), :] = img

    return img_padded

def padding_coord(img, coord, shape_r=224, shape_c=224):
    original_shape = img.shape
    rows_rate = original_shape[0]/shape_r
    cols_rate = original_shape[1]/shape_c
    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * shape_r) // original_shape[0]
        coord[:, 0] = (coord[:, 0] * shape_r) // original_shape[0]
        coord[:, 0] = coord[:, 0] + ((shape_c - new_cols) // 2)
        coord[:, 1] = (coord[:, 1] / rows_rate).astype('int32')
    else:
        new_rows = (original_shape[0] * shape_c) // original_shape[1]
        coord[:, 1] = (coord[:, 1] * shape_c) // original_shape[1]
        coord[:, 1] = coord[:, 1] + ((shape_r - new_rows) // 2)
        coord[:, 0] = (coord[:, 0] / cols_rate).astype('int32')
    return coord

def resize_fixation(img, rows=480, cols=640):
    out = np.zeros((rows, cols))
    factor_scale_r = rows / img.shape[0]
    factor_scale_c = cols / img.shape[1]

    coords = np.argwhere(img)
    for coord in coords:
        r = int(np.round(coord[0]*factor_scale_r))
        c = int(np.round(coord[1]*factor_scale_c))
        if r == rows:
            r -= 1
        if c == cols:
            c -= 1
        out[r, c] = 1

    return out


def padding_fixation(img, shape_r=480, shape_c=640):
    img_padded = np.zeros((shape_r, shape_c))

    original_shape = img.shape
    rows_rate = original_shape[0]/shape_r
    cols_rate = original_shape[1]/shape_c

    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * shape_r) // original_shape[0]
        img = resize_fixation(img, rows=shape_r, cols=new_cols)
        if new_cols > shape_c:
            new_cols = shape_c
        img_padded[:, ((img_padded.shape[1] - new_cols) // 2):((img_padded.shape[1] - new_cols) // 2 + new_cols)] = img
    else:
        new_rows = (original_shape[0] * shape_c) // original_shape[1]
        img = resize_fixation(img, rows=new_rows, cols=shape_c)
        if new_rows > shape_r:
            new_rows = shape_r
        img_padded[((img_padded.shape[0] - new_rows) // 2):((img_padded.shape[0] - new_rows) // 2 + new_rows), :] = img

    return img_padded


def preprocess_images(paths, shape_r, shape_c):
    ims = np.zeros((len(paths), shape_r, shape_c, 3))

    for i, path in enumerate(paths):
        original_image = cv2.imread(path)
        padded_image = padding(original_image, shape_r, shape_c, 3)
        ims[i] = padded_image

    ims[:, :, :, 0] -= 103.939
    ims[:, :, :, 1] -= 116.779
    ims[:, :, :, 2] -= 123.68
    ims = ims.transpose((0, 3, 1, 2))

    return ims


def preprocess_maps(paths, shape_r, shape_c):
    ims = np.zeros((len(paths), 1, shape_r, shape_c))

    for i, path in enumerate(paths):
        original_map = cv2.imread(path, 0)
        padded_map = padding(original_map, shape_r, shape_c, 1)
        ims[i, 0] = padded_map.astype(np.float32)
        ims[i, 0] /= 255.0

    return ims


def preprocess_fixmaps(paths, shape_r, shape_c):
    ims = np.zeros((len(paths), 1, shape_r, shape_c))

    for i, path in enumerate(paths):
        fix_map = scipy.io.loadmat(path)["I"]
        ims[i, 0] = padding_fixation(fix_map, shape_r=shape_r, shape_c=shape_c)

    return ims


def preprocess_mit_fixmaps(paths, shape_r, shape_c):
    ims = np.zeros((len(paths), 1, shape_r, shape_c))

    for i, path in enumerate(paths):
        fix_map = scipy.io.loadmat(path)["fixLocs"]
        ims[i, 0] = padding_fixation(fix_map, shape_r=shape_r, shape_c=shape_c)

    return ims


def preprocess_cat2000_fixmaps(paths, shape_r, shape_c):
    ims = np.zeros((len(paths), 1, shape_r, shape_c))

    for i, path in enumerate(paths):
        fix_map = scipy.io.loadmat(path)["fixLocs"]
        ims[i, 0] = padding_fixation(fix_map, shape_r=shape_r, shape_c=shape_c)

    return ims


def postprocess_predictions(pred, shape_r, shape_c):
    predictions_shape = pred.shape
    rows_rate = shape_r / predictions_shape[0]
    cols_rate = shape_c / predictions_shape[1]

    pred = pred / np.max(pred) * 255

    if rows_rate > cols_rate:
        new_cols = (predictions_shape[1] * shape_r) // predictions_shape[0]
        pred = cv2.resize(pred, (new_cols, shape_r))
        img = pred[:, ((pred.shape[1] - shape_c) // 2):((pred.shape[1] - shape_c) // 2 + shape_c)]
    else:
        new_rows = (predictions_shape[0] * shape_c) // predictions_shape[1]
        pred = cv2.resize(pred, (shape_c, new_rows))
        img = pred[((pred.shape[0] - shape_r) // 2):((pred.shape[0] - shape_r) // 2 + shape_r), :]

    img = scipy.ndimage.filters.gaussian_filter(img, sigma=5)
    img = img / np.max(img) * 255

    return img

if __name__ == '__main__':
    ORIGIN_RESOLUTION_IMG_SET = 'imgs.npy'
    OUTPUT_IMG_SET = 'imgs_384_512.npy'
    SCANPATH_FILE = 'ASD_origin_scanpath.npy'
    OUTPUT_SCANPATH_FILE = 'ASD_384_512_scanpath.npy'
    CHANNELS = 3
    NUM_STEPS = 10

    TO_ROW = 384
    TO_COL = 512

    scanpath = np.load(os.path.join(DATA_DIR, SCANPATH_FILE), encoding = 'latin1')
    imgs = np.load(os.path.join(DATA_DIR, ORIGIN_RESOLUTION_IMG_SET), encoding = 'latin1')

    coord = []
    output_img_set = []

    i = 0
    for img_origin in imgs:
        img = padding(img_origin, TO_ROW, TO_COL, CHANNELS)
        output_img_set.append(img)
        coord_img = []
        for scanpath_p in scanpath[i]:
            scanpath_p = np.array(scanpath_p)
            scanpath_p = padding_coord(img_origin, scanpath_p,
                    TO_ROW, TO_COL)
            if len(scanpath_p) < NUM_STEPS:
                a = np.zeros((NUM_STEPS, len(scanpath_p[0])))
                a[0: len(scanpath_p), :] = scanpath_p
                scanpath_p = a
            else:
                scanpath_p = scanpath_p[0: NUM_STEPS, :]
            scanpath_p = scanpath_p.astype('float32')
            scanpath_p[:, 0] = scanpath_p[:, 0] / TO_COL
            scanpath_p[:, 1] = scanpath_p[:, 1] / TO_ROW
            coord_img.append(scanpath_p)
        coord.append(coord_img)
        i += 1
    coord = np.array(coord)
    output_img_set = np.array(output_img_set)
    #np.save(os.path.join(DATA_DIR, OUTPUT_IMG_SET), output_img_set)
    np.save(os.path.join(DATA_DIR, OUTPUT_SCANPATH_FILE), coord)
