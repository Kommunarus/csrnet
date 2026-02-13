import json
import random
import glob

dir = '/home/neptun/PycharmProjects/CSRNet-pytorch/ds/balka'

files = glob.glob(dir + '/images/*.*')
N = len(files)

all_indx = list(range(N))

random.seed(0)
test_indx = random.sample(all_indx, int(N*0.2), )

train_list = []
test_list = []
for inx in all_indx:
    if inx in test_indx:
        test_list.append(files[inx])
    else:
        train_list.append(files[inx])

with open('./train_list.json', 'w') as f:
    json.dump(train_list, f)

with open('./test_list.json', 'w') as f:
    json.dump(test_list, f)