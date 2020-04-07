import pandas as pd
import shutil
import os
import cv2
import numpy as np
from visualize import grid_transform
from sklearn.feature_extraction import image

df = pd.read_csv('test_clusters_2.csv')
num_rows = len(df)
for label_num in range(11):
    df_label = df['cluster'] == label_num
    df_0 = df[df_label].drop_duplicates(['path', 'cluster'])
    print(len(df_0))

    for row in df_0.iterrows():
        shutil.copy(row[1]['path'], f'colors_labels_2/{label_num}/')
        imagen = cv2.imread(row[1]['path'])
        patch = image.extract_patches_2d(imagen, (32, 32))[row[1]['patch_index']]
        cv2.imwrite(f'colors_labels_2/{label_num}/patch_{row[1]["path"].split(sep="/")[7]}', patch)

    images = os.listdir(f'colors_labels_2/{label_num}/')
    arrs = []
    for ima in images:
        if ima.endswith('.jpg') and ('patch' not in ima):
            bgr = cv2.imread(f'colors_labels_2/{label_num}/' + ima)
            bgr_resized = cv2.resize(bgr, (128, 128), interpolation=cv2.INTER_CUBIC)
            arrs.append(bgr_resized)

    data = np.array(arrs)
    print(data.shape)
    x = grid_transform(data, [128, 128, 3])

    cv2.imwrite(f'colors_labels_2/cluster_images/cluster_{label_num}.jpg', x)

    patches = os.listdir(f'colors_labels_2/{label_num}/')
    arrs = []
    for ima in images:
        if ima.endswith('.jpg') and ('patch' in ima):
            bgr = cv2.imread(f'colors_labels_2/{label_num}/' + ima)
            arrs.append(bgr)

    data = np.array(arrs)
    print(data.shape)
    x = grid_transform(data, [32, 32, 3])

    cv2.imwrite(f'colors_labels_2/cluster_images/cluster_patches_{label_num}.jpg', x)
