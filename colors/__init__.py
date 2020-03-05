import cv2
import numpy as np
import pandas as pd
import random
from math import floor

class DataSampler(object):
    def __init__(self):
        self.shape = [32, 32, 3]
        self.dataset_path = '/content/ClusterGAN/colors/google_colors'
        self.df = pd.read_csv('/content/ClusterGAN/colors/dataset.csv')

    def load_label_names(self):
        return ['black', 'blue', 'brown', 'green', 'grey', 'orange', 'pink', 'purple', 'red', 'white', 'yellow']

    def load_image(self, path):
        bgr = cv2.imread(path)
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
        img = np.reshape(img, 32 * 32 * 3)
        img = img / 255
        return img

    def train(self, batch_size, label=False):
        train_df = self.df['train'] == 1
        first = True
        for label_name in self.load_label_names():
            # leer del csv donde label sea x y este en train, coger n aleatorias
            label_df = self.df['label'] == label_name
            df = self.df[train_df & label_df]
            nums = np.random.randint(low=0, high=len(df), size=floor(batch_size / len(self.load_label_names())))
            df = df.iloc[nums]
            for row in df.iterrows():
                img = self.load_image(self.dataset_path + row[1]['path'])
                img_label = row[1]['label']
                if first:
                    batch = img
                    labels = np.array([img_label])
                    first = False
                else:
                    batch = np.vstack((batch, img))
                    labels = np.append(labels, img_label)
        while batch.shape[0] < 64:
            label_df = self.df['label'] == random.choice(self.load_label_names())
            df = self.df[train_df & label_df]
            df = df.iloc[np.random.randint(low=0, high=len(df))]
            for row in df.iterrows():
                img = self.load_image(self.dataset_path + row[1]['path'])
                img_label = row[1]['label']
                batch = np.vstack((batch, img))
                labels = np.append(labels, img_label)
        if label:
            return batch, labels
        else:
            return batch

    def test(self):
        test_df = self.df['train'] == 0
        df = self.df[test_df]
        first = True
        for row in df.iterrows():
            img = self.load_image(self.dataset_path + row[1]['path'])
            img_label = row[1]['label']
            if first:
                batch = img
                labels = np.array([img_label])
                first = False
            else:
                batch = np.vstack((batch, img))
                labels = np.append(labels, img_label)
        return batch, labels

    def validation(self):
        test_df = self.df['train'] == 0
        df = self.df[test_df]
        first = True
        for row in df.iterrows():
            img = self.load_image(self.dataset_path + row[1]['path'])
            img_label = row[1]['label']
            if first:
                batch = img
                labels = np.array([img_label])
                first = False
            else:
                batch = np.vstack((batch, img))
                labels = np.append(labels, img_label)
        return batch, labels

    def data2img(self, data):
        #                        batch size       [32,32,3]
        return np.reshape(data, [data.shape[0]] + self.shape)

    def load_all(self):
        pass

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
