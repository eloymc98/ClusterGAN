import cv2
import numpy as np
import pandas as pd
import random
from math import floor
from util import load_google_colors


class DataSampler(object):
    def __init__(self):
        self.shape = [32, 32, 3]
        self.count = 0
        self.dataset_path = '/content/ClusterGAN/colors/google_colors'
        # self.df = pd.read_csv('/content/ClusterGAN/colors/dataset.csv')
        # np array de shape (len(dataset), 32*32*3)
        data, labels = load_google_colors()
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
            labels = self.train_labels[(self.count - 1) * batch_size:batch_size * self.count]
        else:
            features1 = self.train_data[(self.count - 1) * batch_size:]
            labels1 = self.train_labels[(self.count - 1) * batch_size:]
            f2 = self.train_data[:batch_size * self.count - self.train_data.shape[0]]
            l2 = self.train_labels[:batch_size * self.count - self.train_data.shape[0]]
            features = np.vstack((features1, f2))
            labels = np.append(labels1, l2)
            self.count = 0

        if label:
            return features, labels
        else:
            return features

    def test(self):
        return self.test_data, self.test_labels

    def validation(self):
        return self.val_data, self.val_labels

    def data2img(self, data):
        #                        batch size       [32,32,3]
        return np.reshape(data, [data.shape[0]] + self.shape)

    def load_all_train(self):
        return self.train_data, self.train_labels

# class NoiseSampler(object):
#
#     def __init__(self, z_dim = 100, mode='uniform'):
#         self.mode = mode
#         self.z_dim = z_dim
#         self.K = 10
#
#         if self.mode == 'mix_gauss':
#             self.mu_mat = (1.0) * np.eye(self.K, self.z_dim)
#             self.sig = 0.1
#
#         elif self.mode == 'one_hot':
#             self.mu_mat = (1.0) * np.eye(self.K)
#             self.sig = 0.10
#
#
#         elif self.mode == 'pca_kmeans':
#
#             data_x = mnist.train.images
#             feature_mean = np.mean(data_x, axis = 0)
#             data_x -= feature_mean
#             data_embed = PCA(n_components=self.z_dim, random_state=0).fit_transform(data_x)
#             data_x += feature_mean
#             kmeans = KMeans(n_clusters=self.K, random_state=0)
#             kmeans.fit(data_embed)
#             self.mu_mat = kmeans.cluster_centers_
#             shift = np.min(self.mu_mat)
#             scale = np.max(self.mu_mat - shift)
#             self.mu_mat = (self.mu_mat - shift)/scale
#             self.sig = 0.15
#
#
#     def __call__(self, batch_size, z_dim):
#         if self.mode == 'uniform':
#             return np.random.uniform(-1.0, 1.0, [batch_size, z_dim])
#         elif self.mode == 'normal':
#             return 0.15*np.random.randn(batch_size, z_dim)
#         elif self.mode == 'mix_gauss':
#             k = np.random.randint(low = 0, high = self.K, size=batch_size)
#             return self.sig*np.random.randn(batch_size, z_dim) + self.mu_mat[k]
#         elif self.mode == 'pca_kmeans':
#             k = np.random.randint(low=0, high=self.K, size=batch_size)
#             return self.sig * np.random.randn(batch_size, z_dim) + self.mu_mat[k]
#         elif self.mode == 'one_hot':
#             k = np.random.randint(low=0, high=self.K, size=batch_size)
#             return np.hstack((self.sig * np.random.randn(batch_size, z_dim-self.K), self.mu_mat[k]))
