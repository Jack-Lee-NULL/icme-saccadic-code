#
#
# file: merge_fixations.py
# author: Jingquan Lee
# date: 2019-04-29
#
#

import numpy as np

def merge_fixations(scanpath, radius):
    s = []
    for i in range(np.shape(scanpath)[0]-1):
        coord_pre = scanpath[i, 0: 2]
        coord_pos = scanpath[i+1, 0: 2]
        r = np.sum(np.power(coord_pre - coord_pos, 2))
        if r <= radius**2:
            continue
        else:
            s.append(coord_pre)
    s.append(scanpath[np.shape(scanpath)[0]-1, 0: 2])
    s = np.array(s)
    print(s.shape)
    zeros = np.zeros(np.shape(scanpath))
    zeros[0: np.shape(s)[0], :] = s
    s = zeros
    return s

if __name__ == '__main__':
    scanpath_file = '/data/jqLee/data/ASD_384_512_scanpath_not_normalized.npy'
    output_file = '/data/jqLee/data/ASD_384_512_scanpath_merged_fixations.npy'
    radius = 25

    scanpath = np.load(scanpath_file)
    output = []
    for i in range(np.shape(scanpath)[0]):
        output_1 = []
        for j in range(np.shape(scanpath[i])[0]):
            s = scanpath[i][j][:, 0: 2]
            s = merge_fixations(s, radius=radius)
            output_1.append(s)
        output.append(output_1)
    np.save(output_file, output)
    print('Has been saved to', output_file)
