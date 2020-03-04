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

    def load_label_names(self):
        return ['black', 'blue', 'brown', 'green', 'grey', 'orange', 'pink', 'purple', 'red', 'white', 'yellow']

    def train(self, batch_size, label=False):
        # normalizar valores entre 0 y 1!!!!!!!!!!!!
        features = None
        labels = None

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
