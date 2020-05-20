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
patch_number = 0
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
    plt.imsave('res.png', res)
    patches = image.extract_patches_2d(res, (32, 32))
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
    plt.imsave('prueba.png', patch)
    labels.append(label)
    patch = patch / 255
    patch = np.reshape(patch, 32 * 32 * 3)
    if first:
        dataset = patch
        first = False
    else:
        dataset = np.vstack((dataset, patch))

labels = np.asarray(labels)
np.save('colors_new_test_patches_data_rgb.npy', dataset)
np.save('colors_new_test_patches_labels_rgb.npy', labels)
f.close()

    # if patch_number % 80 == 0:
    #     data = np.array(data)
    #     labels = np.array(labels)
    #     np.save(f'colors_test_clustering_data_{patch_number}.npy', data)
    #     np.save(f'colors_test_clustering_labels_{patch_number}.npy', labels)
    #     data = []
    #     labels = []

# data = np.array(data)
# labels = np.array(labels)
# f.close()
#
# np.save(f'colors_test_clustering_data_{patch_number}.npy', data)
# np.save(f'colors_test_clustering_labels_{patch_number}.npy', labels)
#


# test_1 = np.load('colors_test_clustering_labels_80.npy')
# print(test_1.shape)
# test_2 = np.load('colors_test_clustering_labels_160.npy')
# print(test_2.shape)
#
# test = np.concatenate((test_1, test_2))
# del test_1, test_2
# test_3 = np.load('colors_test_clustering_labels_240.npy')
# test_4 = np.load('colors_test_clustering_labels_320.npy')
# test = np.concatenate((test, test_3))
# del test_3
# test = np.concatenate((test, test_4))
# del test_4
#
# test_5 = np.load('colors_test_clustering_labels_400.npy')
# test = np.concatenate((test, test_5))
# del test_5
# test_6 = np.load('colors_test_clustering_labels_436.npy')
# test = np.concatenate((test, test_6))
# del test_6
#
# np.save('colors_test_clustering_labels.npy', test)
#
# test = np.load('npy files/colors_new_test_patches_data.npy')
# print(test.shape)
# test_data = np.reshape(test, [test.shape[0], 32 * 32 * 3])
# print(test_data.shape)