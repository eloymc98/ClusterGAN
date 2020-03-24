import numpy as np
import cv2
import pandas as pd
from math import floor
import random
from util import load_termisk_reduced


class DataSampler(object):
    def __init__(self):
        self.shape = [28, 28, 1]
        self.count = 0
        self.dataset_path = '/content/ClusterGAN/termisk/termisk_dataset'
        self.labels = ['0', '1', '2', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']
        self.df = pd.read_csv('/content/ClusterGAN/termisk/dataset.csv')
        print(self.df.head())
        data, labels = load_termisk_reduced()
        # split into train, validation, test
        size = data.shape[0]
        test_index = random.sample(range(0, data.shape[0]), floor(size * 0.1))
        self.test_data = data[test_index]
        self.test_labels = labels[test_index]

        data = np.delete(data, test_index, axis=0)
        labels = np.delete(labels, test_index)

        val_index = random.sample(range(0, data.shape[0]), floor(size * 0.1))
        self.val_data = data[val_index]
        self.val_labels = labels[val_index]

        self.train_data = np.delete(data, val_index, axis=0)
        self.train_labels = np.delete(labels, val_index)

        np.random.shuffle(self.train_data)

    def load_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        img = np.reshape(img, 96 * 96)
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


    def test(self):
        return self.test_data, self.test_labels

    def validation(self):
        return self.val_data, self.val_labels

    def data2img(self, data):
        print(f'Data2img: Data shape = {data.shape}, Self shape = {self.shape}')
        return np.reshape(data, [data.shape[0]] + self.shape)

    def load_all(self):
        pass
