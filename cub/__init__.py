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
        self.shape = [94, 94, 3]
        cifar10_dataset_folder_path = './data/cifar-10-batches-py'
        self.count = 0
        with open(cifar10_dataset_folder_path + '/data_batch_1', mode='rb') as file:
            # note the encoding type is 'latin1'
            batch1 = pickle.load(file, encoding='latin1')
        f1 = batch1['data']
        l1 = np.array([batch1['labels']])
        with open(cifar10_dataset_folder_path + '/data_batch_2', mode='rb') as file:
            # note the encoding type is 'latin1'
            batch2 = pickle.load(file, encoding='latin1')
        f2 = batch2['data']
        l2 = np.array([batch2['labels']])
        with open(cifar10_dataset_folder_path + '/data_batch_3', mode='rb') as file:
            # note the encoding type is 'latin1'
            batch3 = pickle.load(file, encoding='latin1')
        f3 = batch3['data']
        l3 = np.array([batch3['labels']])
        with open(cifar10_dataset_folder_path + '/data_batch_4', mode='rb') as file:
            # note the encoding type is 'latin1'
            batch4 = pickle.load(file, encoding='latin1')
        f4 = batch4['data']
        l4 = np.array([batch4['labels']])
        with open(cifar10_dataset_folder_path + '/data_batch_5', mode='rb') as file:
            # note the encoding type is 'latin1'
            batch5 = pickle.load(file, encoding='latin1')
        f5 = batch5['data']
        l5 = np.array([batch5['labels']])
        self.train_features = np.vstack((f1, f2))
        self.train_features = np.vstack((self.train_features, f3))
        self.train_features = np.vstack((self.train_features, f4))
        self.train_labels = np.append(l1, l2)
        self.train_labels = np.append(self.train_labels, l3)
        self.train_labels = np.append(self.train_labels, l4)

        self.val_features = f5
        self.val_labels = l5

    def load_label_names(self):
        return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    def load_cfar10_test(self, cifar10_dataset_folder_path):
        with open(cifar10_dataset_folder_path + '/test_batch', mode='rb') as file:
            # note the encoding type is 'latin1'
            batch = pickle.load(file, encoding='latin1')

        features = batch['data']
        features = features[:] / 255
        labels = np.asarray(batch['labels'])
        features = features.reshape((len(features), 3, 32, 32)).transpose(0, 2, 3, 1)
        features = features.reshape((len(features), 32 * 32 * 3))
        return features, labels

    def train(self, batch_size, label=False):
        self.count += 1
        if batch_size * self.count <= self.train_features.shape[0]:
            features = self.train_features[(self.count - 1) * batch_size:batch_size * self.count]
            labels = self.train_labels[(self.count - 1) * batch_size:batch_size * self.count]
        else:
            features1 = self.train_features[(self.count - 1) * batch_size:]
            labels1 = self.train_labels[(self.count - 1) * batch_size:]
            f2 = self.train_features[:batch_size * self.count - self.train_features.shape[0]]
            l2 = self.train_labels[:batch_size * self.count - self.train_features.shape[0]]
            features = np.vstack((features1, f2))
            labels = np.append(labels1, l2)
            self.count = 0

        features = features[:] / 255
        features = features.reshape((batch_size, 3, 32, 32)).transpose(0, 2, 3, 1)
        features = features.reshape((batch_size, 32 * 32 * 3))
        if label:
            return features, labels
        else:
            return features

    def test(self):
        features, labels = self.load_cfar10_test('./data/cifar-10-batches-py')
        return features, labels

    def validation(self):
        features = self.val_features
        labels = self.val_labels
        features = features[:] / 255
        features = features.reshape((self.val_features.shape[0], 3, 32, 32)).transpose(0, 2, 3, 1)
        features = features.reshape((self.val_features.shape[0], 32 * 32 * 3))
        return features, labels

    def data2img(self, data):
        #                        batch size       [32,32,3]
        return np.reshape(data, [data.shape[0]] + self.shape)

    def load_all(self):
        pass
