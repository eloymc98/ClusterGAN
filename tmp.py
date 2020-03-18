# from pathlib import Path
#
# ima = Path('/Users/eloymarinciudad/Downloads/')
#
# if ima.is_file():
#     print('File')
# if ima.is_dir():
#     print('Dir')

# import cv2
# import numpy as np

# img = cv2.imread('/Users/eloymarinciudad/Downloads/300.0.png', cv2.IMREAD_GRAYSCALE)
# res = cv2.resize(img, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
# res = res.flatten()/255
# latent = np.zeros(shape=(1, 40))
# x = np.zeros(shape=(1, res.shape[0]))
# x[0, :] = res
#
# print(x[0])
# import util
#
# index = util.closest(np.array([[0.3, 0.5, 0.5], [0, 0, 0.3], [0.8, 0.9, 0.2]]), np.array([0.8, 0.8, 0.1]))
# print(index)
# labelsss = np.zeros(shape=(8, 1))
# zhats = np.array([[2.78456683e-25, 1.13403173e-28, 2.55347444e-21, 1.00000000e+00
#                       , 1.50863351e-17, 7.60306297e-26, 2.45780154e-26, 1.14293015e-21
#                       , 1.95183931e-33, 8.89012444e-36],
#                   [1.08963476e-17, 4.00568077e-12, 2.97130726e-04, 6.19733441e-07
#                       , 2.27435709e-10, 9.99482751e-01, 1.29049271e-10, 2.19487396e-04
#                       , 4.47349440e-16, 3.06283323e-15],
#                   [1.44472004e-16, 2.99856798e-19, 5.20813285e-14, 3.97759252e-14
#                       , 3.88611686e-15, 2.10426745e-18, 2.56510990e-15, 7.85813862e-12
#                       , 4.75163589e-11, 1.00000000e+00],
#                   [1.79301420e-27, 8.49859215e-22, 8.25734812e-14, 4.08180131e-18
#                       , 9.49121817e-21, 1.00000000e+00, 1.33559795e-15, 3.30720830e-18
#                       , 3.75142064e-25, 1.81776360e-22],
#                   [2.01650932e-27, 1.03082570e-17, 1.78927951e-24, 1.74454154e-15
#                       , 1.00000000e+00, 3.77784699e-24, 4.30003968e-18, 3.93773716e-21
#                       , 8.15673625e-27, 4.58766904e-31],
#                   [2.57019536e-21, 3.97949358e-27, 7.33718703e-20, 2.23680866e-22
#                       , 4.22969468e-25, 3.24894860e-26, 7.21501650e-25, 1.47053441e-26
#                       , 1.00000000e+00, 6.51220103e-20],
#                   [9.28357316e-34, 1.92659722e-35, 2.80881774e-24, 1.00000000e+00
#                       , 6.73518018e-23, 8.15417090e-38, 1.95591069e-31, 1.12716567e-28
#                       , 2.21590729e-30, 0.00000000e+00],
#                   [1.04932895e-19, 3.38904427e-20, 3.01676636e-18, 3.98609977e-21
#                       , 1.28237738e-20, 1.01557589e-25, 5.24041828e-25, 5.87892852e-21
#                       , 1.00000000e+00, 8.37501629e-20]])
# print(np.argmax(zhats, axis=1).shape)
# labelsss[np.arange(0,8), 0] = np.argmax(zhats, axis=1)
# print(labelsss)


# import os
# import shutil
#
#
# def copytree(src, dst, symlinks=False, ignore=None):
#     for item in os.listdir(src):
#         print(item)
#         s = os.path.join(src, item)
#         print(s)
#         d = os.path.join(dst, item)
#         print(d)
#         if os.path.isdir(s):
#             shutil.copytree(src=s, dst=d, symlinks=symlinks, ignore=ignore)
#         else:
#             shutil.copy2(src=s, dst=dst)
#
#
# copytree('/Users/eloymarinciudad/Downloads/prueba', '/Users/eloymarinciudad/Downloads')

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
# img = cv2.imread('/Users/eloymarinciudad/Downloads/20190625R15S96/train/2/26837_240_0_0.png', cv2.IMREAD_GRAYSCALE)
# img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
# print(img.shape)
# #print(img)
#
# img = np.reshape(img, 28 * 28)
# print(img.shape)
# img = img / 255
# print(img)
#
# batch = np.ndarray(shape=(64, 28 * 28), dtype=float)
#
# batch[0] = img
# print(batch)


# import pandas as pd
# import numpy as np
# import cv2
#
# # test: 95 imagenes por clase
#
# df = pd.read_csv('termisk/dataset.csv')
#
# df1 = df['label'] == 0
# df2 = df['label'] == 1
# df3 = df['label'] == 2
# df4 = df['label'] == 4
# df5 = df['label'] == 5
# df6 = df['label'] == 6
# df7 = df['label'] == 7
# df8 = df['label'] == 8
# df9 = df['label'] == 9
# df10 = df['label'] == 10
# df11 = df['label'] == 11
# df12 = df['label'] == 12
# df13 = df['label'] == 13
# df14 = df['label'] == 14
# df15 = df['label'] == 15
# df16 = df['label'] == 16
# print(f'Total images: {len(df)}')
# print(f'Label 0 size: {len(df[df1])}')
# print(f'Label 1 size: {len(df[df2])}')
# print(f'Label 2 size: {len(df[df3])}')
# print(f'Label 4 size: {len(df[df4])}')
# print(f'Label 5 size: {len(df[df5])}')
# print(f'Label 6 size: {len(df[df6])}')
# print(f'Label 7 size: {len(df[df7])}')
# print(f'Label 8 size: {len(df[df8])}')
# print(f'Label 9 size: {len(df[df9])}')
# print(f'Label 10 size: {len(df[df10])}')
# print(f'Label 11 size: {len(df[df11])}')
# print(f'Label 12 size: {len(df[df12])}')
# print(f'Label 13 size: {len(df[df13])}')
# print(f'Label 14 size: {len(df[df14])}')
# print(f'Label 15 size: {len(df[df15])}')
# print(f'Label 16 size: {len(df[df16])}')

import os
import cv2
import numpy as np
from math import floor


def load_termisk_reduced():
    path = "/Users/eloymarinciudad/Downloads/termisk_dataset"
    split_paths = os.listdir(path)
    print(split_paths)
    labels = []

    first = True
    for split in split_paths:
        subdir = path + '/' + split
        print(subdir)
        if os.path.isdir(subdir):
            classes = os.listdir(subdir)
            for label in classes:
                class_path = subdir + '/' + label
                if label in ('5', '6', '8', '10', '11'):
                    print(label)
                    for image in os.listdir(class_path):
                        if os.path.isfile(class_path + '/' + image) and image.endswith('.png'):
                            if label == '8':
                                index_label = 2
                            elif label in ('10', '11'):
                                index_label = int(label) - 7
                            else:
                                index_label = int(label) - 5
                            try:
                                img = cv2.imread(class_path + '/' + image, cv2.IMREAD_GRAYSCALE)
                                # img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
                                img = np.reshape(img, 96 * 96)
                                img = img / 255
                            except ValueError:
                                print(class_path + '/' + image)
                            labels.append(index_label)
                            if first:
                                dataset = img
                                first = False
                            else:
                                dataset = np.vstack((dataset, img))
    labels = np.asarray(labels)
    return dataset, labels

import random
data, labels = load_termisk_reduced()
# test_index = np.random.randint(low=0, high=data.shape[0], size=floor(data.shape[0] * 0.1))
size = data.shape[0]
test_index = random.sample(range(0, data.shape[0]), floor(size * 0.1))

print(data.shape)
print(data.shape[0])
test_data = data[test_index]
test_labels = labels[test_index]
print(len(test_index))
print(test_data.shape)

data = np.delete(data, test_index, axis=0)
labels = np.delete(labels, test_index)
print(data.shape)
print(labels.shape)
print(test_index)

val_index = random.sample(range(0, data.shape[0]), floor(size * 0.1))
val_data = data[val_index]
val_labels = labels[val_index]
print(val_data.shape)

data = np.delete(data, val_index, axis=0)
labels = np.delete(labels, val_index)
print(data.shape)
print(labels.shape)