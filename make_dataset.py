import PIL.Image as Image
import xmltodict
import os
import glob
import numpy as np
import scipy
import scipy.io as io
from matplotlib import pyplot as plt
from matplotlib import cm as CM
import h5py
from scipy.ndimage import gaussian_filter

def gaussian_filter_density(gt):
    print(gt.shape)
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    # pts = np.array(zip(np.nonzero(gt)[1], np.nonzero(gt)[0]))
    pts = np.column_stack((np.nonzero(gt)[1], np.nonzero(gt)[0]))
    leafsize = 200
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    k = min(4, gt_count)
    distances, locations = tree.query(pts, k=k)

    print('generate density...')
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1],pt[0]] = 1.
        if gt_count > 1:
            # Берём только конечные расстояния
            valid_dists = [d for d in distances[i][1:] if np.isfinite(d)]
            sigma = sum(valid_dists[:3]) * 0.05

        else:
            sigma = np.average(np.array(gt.shape)) / 10. / 10.
        # if gt_count > 1:
        #     sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        # else:
        #     sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point
        density += gaussian_filter(pt2d, sigma, mode='constant')
    print('done.')
    return density

dir_ds = './ds/balka'

with open(os.path.join(dir_ds, 'annotations.xml'), 'r', encoding='utf-8') as f:
    data_raw = xmltodict.parse(f.read())

data = {row['@name']: [p['@points'] for p in row['points']] for row in data_raw['annotations']['image']}

img_paths = []
for img_path in glob.glob(os.path.join(dir_ds, 'images', '*.*')):
    img_paths.append(img_path)


for img_path in img_paths:
    print(img_path)
    bn = os.path.basename(img_path)
    # mat = io.loadmat(img_path.replace('.jpg','.mat').replace('images','ground_truth').replace('IMG_','GT_IMG_'))
    img= plt.imread(img_path)
    k = np.zeros((img.shape[0],img.shape[1]))
    gt = data[bn]
    for i in range(0,len(gt)):
        point = gt[i]
        point = point.split(',')
        x = int(float(point[0]))
        y = int(float(point[1]))
        k[y, x]=1
    k = gaussian_filter_density(k)
    n, _  = os.path.splitext(bn)
    with h5py.File(os.path.join(dir_ds, 'ground_truth', f'{n}.h5'), 'w') as hf:
            hf['density'] = k

plt.imshow(Image.open(img_paths[0]))
plt.show()

name0 = img_paths[0]
bn = os.path.basename(name0)
n, _ = os.path.splitext(bn)
f_h5 = os.path.join(dir_ds, 'ground_truth', f'{n}.h5')

gt_file = h5py.File(f_h5,'r')
groundtruth = np.asarray(gt_file['density'])
plt.imshow(groundtruth,cmap=CM.jet)
plt.show()

print(np.sum(groundtruth))