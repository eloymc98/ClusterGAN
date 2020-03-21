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
        # self.dataset_path = '/content/ClusterGAN/colors_new'
        # self.df = pd.read_csv('/content/ClusterGAN/colors/dataset.csv')
        # np array de shape (len(dataset), 32*32*3)
        first = True
        count = 0
        for i in range(11):
            r = i * 0.1
            for j in range(11):
                g = j * 0.1
                for k in range(11):
                    b = k * 0.1
                    ima = np.array([[[r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b]],
                                    [[r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b]],
                                    [[r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b]],
                                    [[r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b]],
                                    [[r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b]],
                                    [[r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b]],
                                    [[r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b]],
                                    [[r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b]],
                                    [[r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b]],
                                    [[r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b]],
                                    [[r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b]],
                                    [[r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b]],
                                    [[r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b]],
                                    [[r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b]],
                                    [[r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b]],
                                    [[r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b]],
                                    [[r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b]],
                                    [[r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b]],
                                    [[r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b]],
                                    [[r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b]],
                                    [[r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b]],
                                    [[r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b]],
                                    [[r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b]],
                                    [[r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b]],
                                    [[r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b]],
                                    [[r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b]],
                                    [[r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b]],
                                    [[r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b]],
                                    [[r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b]],
                                    [[r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b]],
                                    [[r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b]],
                                    [[r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b],
                                     [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                                     [r, g, b]]])
                    ima = np.reshape(ima, 32 * 32 * 3)
                    if first:
                        self.train_data = ima
                        first = False
                    else:
                        self.train_data = np.vstack((self.train_data, ima))

                    count += 1
                    if count % 100 == 0:
                        print(count)

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

    def test(self):
        pass

    def validation(self):
        pass

    def data2img(self, data):
        #                        batch size       [32,32,3]
        return np.reshape(data, [data.shape[0]] + self.shape)

    def load_all(self):
        pass
