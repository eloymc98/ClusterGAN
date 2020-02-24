import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import cv2
import pandas as pd


class DataSampler(object):
    def __init__(self):
        self.shape = [28, 28, 1]
        self.dataset_path = '/content/ClusterGAN/termisk/termisk_dataset'
        self.labels = ['0', '1', '2', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']
        self.df = pd.read_csv('/content/ClusterGAN/termisk/dataset.csv')
        print(self.df.head())

    def load_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        img = np.reshape(img, 28 * 28)
        img = img / 255
        return img

    def train(self, batch_size, label=False):
        train_df = self.df['train'] == 1
        first = True
        for label_num in self.labels:
            # leer del csv donde label sea x y este en train, coger n aleatorias
            label_df = self.df['label'] == int(label_num)
            df = self.df[train_df & label_df]
            nums = np.random.randint(low=0, high=len(df), size=round(batch_size / len(self.labels)))
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
        if label:
            return batch, labels
        else:
            return batch

    def test(self):
        test_df = self.df['train'] == 0.5
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
        validation_df = self.df['train'] == 0
        df = self.df[validation_df]
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
        print(f'Data2img: Data shape = {data.shape}, Self shape = {self.shape}')
        return np.reshape(data, [data.shape[0]] + self.shape)

    def load_all(self):
        pass
