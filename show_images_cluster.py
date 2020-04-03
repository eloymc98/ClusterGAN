import pandas as pd
import shutil
import os
import cv2
import numpy as np
from visualize import grid_transform

df = pd.read_csv('test_clusters.csv')
num_rows = len(df)
for label_num in range(11):
    # label_0 = df['cluster'] == label_num
    # df_0 = df[label_0].drop_duplicates()
    # print(len(df_0))
    #
    # for row in df_0.iterrows():
    #     shutil.copy(row[1]['path'], f'colors_labels/{label_num}/')

    images = os.listdir(f'colors_labels/{label_num}/')
    arrs = []
    for ima in images:
        if ima.endswith('.jpg'):
            bgr = cv2.imread(f'colors_labels/{label_num}/' + ima)
            bgr_resized = cv2.resize(bgr, (128, 128), interpolation=cv2.INTER_CUBIC)
            arrs.append(bgr_resized)

    data = np.array(arrs)
    print(data.shape)
    x = grid_transform(data, [128, 128, 3])

    cv2.imwrite(f'colors_labels/cluster_images/cluster_{label_num}.jpg', x)
