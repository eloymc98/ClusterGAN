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

# import os
# import cv2
# import numpy as np
# from math import floor
#
#
# def load_termisk_reduced():
#     path = "/Users/eloymarinciudad/Downloads/termisk_dataset"
#     split_paths = os.listdir(path)
#     print(split_paths)
#     labels = []
#
#     first = True
#     for split in split_paths:
#         subdir = path + '/' + split
#         print(subdir)
#         if os.path.isdir(subdir):
#             classes = os.listdir(subdir)
#             for label in classes:
#                 class_path = subdir + '/' + label
#                 if label in ('5', '6', '8', '10', '11'):
#                     print(label)
#                     count = 0
#                     for image in os.listdir(class_path):
#
#                         if os.path.isfile(class_path + '/' + image) and image.endswith('.png'):
#                             if label == '8':
#                                 index_label = 2
#                             elif label in ('10', '11'):
#                                 index_label = int(label) - 7
#                             else:
#                                 index_label = int(label) - 5
#
#                             img = cv2.imread(class_path + '/' + image, cv2.IMREAD_GRAYSCALE)
#                             count += 1
#                             if count % 100 == 0:
#                                 print(count)
#                             # img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
#                             img = np.reshape(img, 96 * 96)
#                             img = img / 255
#                             labels.append(index_label)
#                             if first:
#                                 dataset = img
#                                 first = False
#                             else:
#                                 dataset = np.vstack((dataset, img))
#     labels = np.asarray(labels)
#     return dataset, labels
#
#
# import random
#
# data, labels = load_termisk_reduced()
# # test_index = np.random.randint(low=0, high=data.shape[0], size=floor(data.shape[0] * 0.1))
# size = data.shape[0]
# test_index = random.sample(range(0, data.shape[0]), floor(size * 0.1))
#
# print(data.shape)
# print(data.shape[0])
# test_data = data[test_index]
# test_labels = labels[test_index]
# print(len(test_index))
# print(test_data.shape)
#
# data = np.delete(data, test_index, axis=0)
# labels = np.delete(labels, test_index)
# print(data.shape)
# print(labels.shape)
# print(test_index)
#
# val_index = random.sample(range(0, data.shape[0]), floor(size * 0.1))
# val_data = data[val_index]
# val_labels = labels[val_index]
# print(val_data.shape)
#
# data = np.delete(data, val_index, axis=0)
# labels = np.delete(labels, val_index)
# print(data.shape)
# print(labels.shape)
# print('-----------------------------\n')
# print(data[0])
# np.random.shuffle(data)
# print(data.shape)
# print(data[0])


# from sklearn.feature_extraction import image
# import cv2
# import random
# import numpy as np
#
# ima = cv2.imread('/Users/eloymarinciudad/Downloads/colors_new_original/orange/00000015.jpg')
# print(ima.dtype)
#
# cv2.imwrite('ima.jpg', ima)
# ima = ima[:, :, [2, 1, 0]]
# # patches = image.extract_patches_2d(ima, (32, 32))
# print(ima.shape)
# # (filas, columnas, canales)
# n = ima.shape[0]
# m = ima.shape[1]
# mid_ima = ima[int(n / 2 - n / 4): int(n / 2 + n / 4), int(m / 2 - m / 4): int(m / 2 + m / 4)]
# cv2.imwrite('mid_ima.jpg', mid_ima)
#
# patches = image.extract_patches_2d(mid_ima, (32, 32))
# random_index = random.randrange(len(patches))
# patch = patches[random_index]
# patch_bgr = patch[:, :, [2, 1, 0]]
# cv2.imwrite('patch.jpg', patch_bgr)
# mid_patches = patches[intx(len(patches) / 2 - 500): int(len(patches) / 2 + 500)]
# # patch = random.choice(mid_patches)
# random_index = random.randrange(len(mid_patches))
# patch = mid_patches[random_index]
# patch_bgr = patch[:, :, [2, 1, 0]]
# cv2.imwrite('patch.jpg', patch_bgr)
# print(patch_bgr.dtype)
# patch_lab = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB)
# print(patch_lab.dtype)
# print(max(np.reshape(patch_lab, 32 * 32 * 3)))
# print(min(np.reshape(patch_lab, 32 * 32 * 3)))
# patch_lab_norm = patch_lab / 255
# print(patch_lab_norm.dtype)
# cv2.imwrite('patch_lab-norm.jpg', patch_lab_norm * 255)
#
# bgr_norm = cv2.cvtColor(patch_lab_nor, cv2.COLOR_LAB2BGR)
# cv2.imwrite('bgr-norm.jpg', bgr_norm)

import os
import numpy as np
import cv2
import shutil
import random
from sklearn.feature_extraction import image


def split_colors_new_dataset():
    path = "/Users/eloymarinciudad/Downloads/colors_new_original"
    classes = os.listdir(path)

    print(classes)
    num_of_images = 0
    for color in classes:
        subdir_class = path + '/' + color
        print(subdir_class)
        if os.path.isdir(subdir_class):
            for imagen in os.listdir(subdir_class):
                shutil.copy(subdir_class + '/' + imagen,
                            f'/Users/eloymarinciudad/Downloads/colors_new/{color}_{imagen}')
                num_of_images += 1

    dest_path = '/Users/eloymarinciudad/Downloads/colors_new'
    imagenes = os.listdir(dest_path)
    random_index_list = random.sample(range(num_of_images), round(num_of_images * 0.2))
    test_list = random_index_list[:round(len(random_index_list) / 2)]
    val_list = random_index_list[round(len(random_index_list) / 2):]
    ima_index = 0
    for imagen in imagenes:
        if imagen.endswith('.jpg'):
            color = imagen.split(sep='_')[0]

            if ima_index in test_list:
                shutil.move(dest_path + '/' + imagen, dest_path + f'/test/{color}/' + imagen)
            elif ima_index in val_list:
                shutil.move(dest_path + '/' + imagen, dest_path + f'/validation/{color}/' + imagen)
            else:
                shutil.move(dest_path + '/' + imagen, dest_path + f'/train/{color}/' + imagen)

            ima_index += 1


def colors_new_train_patches_to_npy_file():
    path = '/Users/eloymarinciudad/Downloads/colors_new/train'
    label = {'black': 0, 'blue': 1, 'brown': 2, 'green': 3, 'grey': 4, 'orange': 5, 'pink': 6,
             'purple': 7, 'red': 8, 'white': 9, 'yellow': 10}

    labels = []
    first = True
    classes = os.listdir(path)
    for color in classes:
        subdir_class = path + '/' + color
        print(subdir_class)
        if os.path.isdir(subdir_class):
            for imagen in os.listdir(subdir_class):
                if os.path.isfile(subdir_class + '/' + imagen) and imagen.endswith('.jpg'):
                    bgr = cv2.imread(subdir_class + '/' + imagen)

                    img = bgr[:, :, [2, 1, 0]]
                    n = img.shape[0]
                    m = img.shape[1]
                    mid_ima = img[int(n / 2 - n / 4): int(n / 2 + n / 4), int(m / 2 - m / 4): int(m / 2 + m / 4)]
                    patches = image.extract_patches_2d(mid_ima, (32, 32))
                    random_index = random.randrange(len(patches))
                    patch = patches[random_index]
                    patch = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB)
                    patch = patch / 255
                    img = np.reshape(patch, 32 * 32 * 3)

                    labels.append(label[color])

                    if first:
                        dataset = img
                        first = False
                    else:
                        dataset = np.vstack((dataset, img))
    labels = np.asarray(labels)
    np.save('colors_new_train_patches_data.npy', dataset)
    np.save('colors_new_train_patches_labels.npy', labels)



data = np.load('colors_new_train_patches_data.npy')
labels = np.load('colors_new_train_patches_labels.npy')

print(data.shape)
print(labels.shape)