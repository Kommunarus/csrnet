import random
import os
from PIL import Image, ImageFilter, ImageDraw
import numpy as np
import h5py
from PIL import ImageStat
import cv2

def load_data(img_path,train = True):
    bn = os.path.basename(img_path)
    n, _ = os.path.splitext(bn)
    dir_ds = '/home/neptun/PycharmProjects/CSRNet-pytorch/ds/balka'
    gt_path = os.path.join(dir_ds, 'ground_truth', f'{n}.h5')

    # img = Image.open(img_path).convert('RGB')
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['density'])

    target = cv2.resize(target,(target.shape[1]//8,target.shape[0]//8), interpolation = cv2.INTER_AREA)*8*8

    return img, target