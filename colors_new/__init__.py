import cv2
import numpy as np
import pandas as pd
import random
from math import floor
from util import load_colors_new


class DataSampler(object):
    def __init__(self):
        self.shape = [32, 32, 3]
        self.count = 0
        self.dataset_path = '/content/ClusterGAN/colors_new'
        # self.df = pd.read_csv('/content/ClusterGAN/colors/dataset.csv')
        # np array de shape (len(dataset), 32*32*3)
        # data, labels = load_colors_new()
        # # split into train, validation, test
        # size = data.shape[0]
        # test_index = random.sample(range(0, data.shape[0]), floor(size * 0.1))
        # self.test_data = data[test_index]
        # self.test_labels = labels[test_index]
        #
        # data = np.delete(data, test_index, axis=0)
        # labels = np.delete(labels, test_index)
        #
        # val_index = random.sample(range(0, data.shape[0]), floor(size * 0.1))
        # self.val_data = data[val_index]
        # self.val_labels = labels[val_index]
        #
        # self.train_data = np.delete(data, val_index, axis=0)
        # self.train_labels = np.delete(labels, val_index)

        # self.train_data = np.load('/content/colors_train_patches_extended_data.npy')
        # self.train_labels = np.load('/content/colors_train_patches_extended_labels.npy')

        #self.train_data = np.load('/content/colors_new_train_patches_data.npy')
        #self.train_labels = np.load('/content/colors_new_train_patches_labels.npy')

        self.train_data = np.load('/content/colors_train_rgb_data.npy')
        self.train_labels = np.load('/content/colors_train_rgb_labels.npy')
        # self.train_data = np.load('/content/colors_train_labimg_data.npy')
        # self.train_labels = np.load('/content/colors_train_labimg_labels.npy')

        self.test_data = np.load('/content/colors_new_test_point_data_rgb.npy')
        # self.test_data = np.reshape(self.test_data, [self.test_data.shape[0], 32 * 32 * 3])
        self.test_labels = np.load('/content/colors_new_test_point_labels_rgb.npy')
        self.test_ima_names = np.load('colors_new_test_point_imanumbers_rgb.npy')
        # self.test_img_names = pd.read_csv('/content/test_patches_2.csv', header=None, names=['path', 'patch_index'])
        np.random.shuffle(self.train_data)

    def load_label_names(self):
        return ['black', 'blue', 'brown', 'green', 'grey', 'orange', 'pink', 'purple', 'red', 'white', 'yellow']

    def load_image(self, path):
        bgr = cv2.imread(path)
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        # img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
        img = np.reshape(img, 32 * 32 * 3)
        img = img / 255
        return img

    def train(self, batch_size, label=False):
        self.count += 1
        if batch_size * self.count <= self.train_data.shape[0]:
            features = self.train_data[(self.count - 1) * batch_size:batch_size * self.count]
            # labels = self.train_labels[(self.count - 1) * batch_size:batch_size * self.count]
        else:
            features1 = self.train_data[(self.count - 1) * batch_size:]
            # labels1 = self.train_labels[(self.count - 1) * batch_size:]
            f2 = self.train_data[:batch_size * self.count - self.train_data.shape[0]]
            # l2 = self.train_labels[:batch_size * self.count - self.train_data.shape[0]]
            features = np.vstack((features1, f2))
            # labels = np.append(labels1, l2)
            self.count = 0

        return features

        # if label:
        #     return features, labels
        # else:
        #     return features

    def test(self, index=False):
        # return self.test_data, self.test_labels, self.test_img_names
        if index:
            return self.test_data, self.test_labels, self.test_ima_names
        else:
            return self.test_data, self.test_labels

    def validation(self):
        return self.val_data, self.val_labels

    def data2img(self, data):
        #                        batch size       [32,32,3]
        return np.reshape(data, [data.shape[0]] + self.shape)

    def load_all(self):
        pass
