import numpy as np
import pickle

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from math import floor


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


class DataSampler(object):
    def __init__(self):
        self.shape = [32, 32, 3]
        self.batch_number = 1
        self.index = 0
        self.count = 0
        cifar10_dataset_folder_path = './data/cifar-10-batches-py'

        with open(cifar10_dataset_folder_path + '/data_batch_1', mode='rb') as file:
            # note the encoding type is 'latin1'
            self.batch1 = pickle.load(file, encoding='latin1')
        with open(cifar10_dataset_folder_path + '/data_batch_2', mode='rb') as file:
            # note the encoding type is 'latin1'
            self.batch2 = pickle.load(file, encoding='latin1')
        with open(cifar10_dataset_folder_path + '/data_batch_3', mode='rb') as file:
            # note the encoding type is 'latin1'
            self.batch3 = pickle.load(file, encoding='latin1')
        with open(cifar10_dataset_folder_path + '/data_batch_4', mode='rb') as file:
            # note the encoding type is 'latin1'
            self.batch4 = pickle.load(file, encoding='latin1')
        with open(cifar10_dataset_folder_path + '/data_batch_5', mode='rb') as file:
            # note the encoding type is 'latin1'
            self.batch5 = pickle.load(file, encoding='latin1')
        self.batch = self.batch1


    def load_label_names(self):
        return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    def load_cfar10_batch(self, cifar10_dataset_folder_path):
        pass

        # # features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
        # # labels = batch['labels']
        # if self.index + batch_size <= floor(len(batch['data'] / batch_size)) * batch_size:
        #     features = batch['data'][self.index:batch_size + self.index]
        #     labels = batch['labels'][self.index:batch_size + self.index]
        #     self.index += batch_size
        # else:
        #     self.batch_number += 1
        #     self.index = 0
        #     with open(cifar10_dataset_folder_path + '/data_batch_' + str(self.batch_number), mode='rb') as file:
        #         # note the encoding type is 'latin1'
        #         batch = pickle.load(file, encoding='latin1')
        #     features = batch['data'][self.index:batch_size + self.index]
        #     labels = batch['labels'][self.index:batch_size + self.index]
        #     self.index += batch_size

        # return batch1, batch2, batch3, batch4, batch5

    def load_cfar10_test(self, cifar10_dataset_folder_path):
        with open(cifar10_dataset_folder_path + '/test_batch', mode='rb') as file:
            # note the encoding type is 'latin1'
            batch = pickle.load(file, encoding='latin1')

        features = batch['data']
        features = features[:]/255
        labels = batch['labels']
        return features, labels

    def train(self, batch_size, label=False):
        # normalizar valores entre 0 y 1!!!!!!!!!!!!
        features = None
        labels = None
        self.count += 1
        if self.count < 157:
            features = self.batch['data'][self.index:batch_size + self.index]
            features = features[:] / 255
            labels = self.batch['labels'][self.index:batch_size + self.index]
        else:

            self.count = 1
            self.index = 0
            self.batch_number += 1
            if self.batch_number == 2:
                self.batch = self.batch2
            elif self.batch_number == 3:
                self.batch = self.batch3
            elif self.batch_number == 4:
                self.batch = self.batch4
            elif self.batch_number == 5:
                self.batch = self.batch5
            elif self.batch_number == 6:
                self.batch = self.batch1
                self.batch_number = 1
            features = self.batch['data'][self.index:batch_size + self.index]
            features = features[:] / 255
            labels = self.batch['labels'][self.index:batch_size + self.index]

        if label:
            return features, labels
        else:
            return features

    def test(self):
        features, labels = self.load_cfar10_test('./data/cifar-10-batches-py')
        return features, labels

    def validation(self):
        features, labels = self.test()
        return features, labels

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
