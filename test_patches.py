import cv2
from sklearn.feature_extraction import image
import random
import re
import numpy as np
import matplotlib.pyplot as plt

x = 'cars/black/'
y = x.split(sep='/')[1]

class_labels = {'black': 0, 'blue': 1, 'brown': 2, 'green': 3, 'grey': 4, 'orange': 5, 'pink': 6,
                'purple': 7, 'red': 8, 'white': 9, 'yellow': 10}

f = open('/Users/eloymarinciudad/Downloads/ebay/test_images.txt')
data = []
labels = []
ima_numbers = []
ima_number = 0
first = True
for line in f:
    line = line.replace('./', '')
    line = line.replace('\n', '')
    img = cv2.imread('/Users/eloymarinciudad/Downloads/ebay/' + line)
    img = img[:, :, [2, 1, 0]]

    path_before_ima = line.split(sep='0')[0]
    print(path_before_ima)
    label_name = path_before_ima.split(sep='/')[1]
    label = class_labels[label_name]
    ima_name = re.findall('[0-9]+', line)[0]
    print(ima_name)
    ima_name = ima_name + '_MASK.png'

    mask = cv2.imread('/Users/eloymarinciudad/Downloads/ebay/' + path_before_ima + ima_name, 0)
    res = cv2.bitwise_and(img, img, mask=mask)
    # plt.imsave('res.png', res)
    patches = image.extract_patches_2d(res, (1, 1))
    for i in range(25):
        random_index = random.randrange(len(patches))
        patch = patches[random_index]
        count = 0
        while min(patch[:, :, 0].flatten()) == 0 and min(patch[:, :, 1].flatten()) == 0 and min(
                patch[:, :, 2].flatten()) == 0:
            random_index = random.randrange(len(patches))
            patch = patches[random_index]
            count += 1
            if count == 10000:
                break
        if count == 10000:
            continue
        # data.append(patch)
        # plt.imsave('prueba.png', patch)
        patch = cv2.resize(patch, dsize=(32, 32), interpolation=cv2.INTER_NEAREST)
        # plt.imsave('prueba2.png', patch)
        labels.append(label)
        ima_numbers.append(ima_number)
        patch = patch / 255
        patch = np.reshape(patch, 32 * 32 * 3)
        if first:
            dataset = patch
            first = False
        else:
            dataset = np.vstack((dataset, patch))
    ima_number += 1

labels = np.asarray(labels)
ima_numbers = np.asarray(ima_numbers)
np.save('colors_new_test_point_data_rgb.npy', dataset)
np.save('colors_new_test_point_labels_rgb.npy', labels)
np.save('colors_new_test_point_imanumbers_rgb.npy', ima_numbers)
f.close()
