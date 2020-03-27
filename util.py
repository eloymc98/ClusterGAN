import numpy as np
import os
import cv2
from sklearn.feature_extraction import image
import random
import csv
import shutil


def colors_new_train_patches_to_npy_file():
    path = '/Users/eloymarinciudad/Downloads/colors_new/train'
    label = {'black': 0, 'blue': 1, 'brown': 2, 'green': 3, 'grey': 4, 'orange': 5, 'pink': 6,
             'purple': 7, 'red': 8, 'white': 9, 'yellow': 10}

    labels = []
    first = True
    classes = os.listdir(path)
    for color in classes:
        subdir_class = path + '/' + color
        print(subdir_class)
        if os.path.isdir(subdir_class):
            for imagen in os.listdir(subdir_class):
                if os.path.isfile(subdir_class + '/' + imagen) and imagen.endswith('.jpg'):
                    bgr = cv2.imread(subdir_class + '/' + imagen)

                    img = bgr[:, :, [2, 1, 0]]
                    n = img.shape[0]
                    m = img.shape[1]
                    mid_ima = img[int(n / 2 - n / 4): int(n / 2 + n / 4), int(m / 2 - m / 4): int(m / 2 + m / 4)]
                    patches = image.extract_patches_2d(mid_ima, (32, 32))
                    random_index = random.randrange(len(patches))
                    patch = patches[random_index]
                    patch = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB)
                    patch = patch / 255
                    img = np.reshape(patch, 32 * 32 * 3)

                    labels.append(label[color])

                    if first:
                        dataset = img
                        first = False
                    else:
                        dataset = np.vstack((dataset, img))
    labels = np.asarray(labels)
    np.save('colors_new_train_patches_data.npy', dataset)
    np.save('colors_new_train_patches_labels.npy', labels)


def split_colors_new_dataset():
    path = "/Users/eloymarinciudad/Downloads/colors_new_original"
    classes = os.listdir(path)

    print(classes)
    num_of_images = 0
    for color in classes:
        subdir_class = path + '/' + color
        print(subdir_class)
        if os.path.isdir(subdir_class):
            for imagen in os.listdir(subdir_class):
                shutil.copy(subdir_class + '/' + imagen,
                            f'/Users/eloymarinciudad/Downloads/colors_new/{color}_{imagen}')
                num_of_images += 1

    dest_path = '/Users/eloymarinciudad/Downloads/colors_new'
    imagenes = os.listdir(dest_path)
    random_index_list = random.sample(range(num_of_images), round(num_of_images * 0.2))
    test_list = random_index_list[:round(len(random_index_list) / 2)]
    val_list = random_index_list[round(len(random_index_list) / 2):]
    ima_index = 0
    for imagen in imagenes:
        if imagen.endswith('.jpg'):
            color = imagen.split(sep='_')[0]

            if ima_index in test_list:
                shutil.move(dest_path + '/' + imagen, dest_path + f'/test/{color}/' + imagen)
            elif ima_index in val_list:
                shutil.move(dest_path + '/' + imagen, dest_path + f'/validation/{color}/' + imagen)
            else:
                shutil.move(dest_path + '/' + imagen, dest_path + f'/train/{color}/' + imagen)

            ima_index += 1


def load_termisk_reduced():
    path = "/content/termisk_dataset"
    # path = "/Users/eloymarinciudad/Downloads/termisk_dataset"
    split_paths = os.listdir(path)
    print(split_paths)
    labels = []
    index_label = 0
    first = True
    for split in split_paths:
        subdir = path + '/' + split
        print(subdir)
        if os.path.isdir(subdir):
            classes = os.listdir(subdir)
            classes = [s for s in classes if '.' not in s]
            print(classes)
            for label in classes:
                class_path = subdir + '/' + label
                count = 0
                if label not in ('17', '3', '0', '1', '2', '4', '7', '9', '12', '13', '14', '16'):
                    for image in os.listdir(class_path):
                        if os.path.isfile(class_path + '/' + image) and image.endswith('.png'):
                            img = cv2.imread(class_path + '/' + image, cv2.IMREAD_GRAYSCALE)
                            img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
                            img = np.reshape(img, 64 * 64)
                            img = img / 255
                            labels.append(index_label)
                            if first:
                                dataset = img
                                first = False
                            else:
                                dataset = np.vstack((dataset, img))
                            count += 1
                        if count >= 1650:
                            break
                    index_label += 1
    labels = np.asarray(labels)
    return dataset, labels


def load_colors_new():
    path = "/content/colors_new"
    classes = os.listdir(path)
    logs_file = open('/content/ClusterGAN/color_patches_logs.csv', 'w')
    writer = csv.writer(logs_file)
    writer.writerow(['ima_path', 'patch_index'])
    print(classes)
    labels = []
    index_label = 0
    first = True
    for color in classes:
        subdir_class = path + '/' + color
        print(subdir_class)
        if os.path.isdir(subdir_class):
            for imagen in os.listdir(subdir_class):
                if os.path.isfile(subdir_class + '/' + imagen):
                    bgr = cv2.imread(subdir_class + '/' + imagen)
                    # bgr = cv2.resize(bgr, (128, 128), interpolation=cv2.INTER_AREA)
                    # img = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
                    img = bgr[:, :, [2, 1, 0]]
                    n = img.shape[0]
                    m = img.shape[1]
                    mid_ima = img[int(n / 2 - n / 4): int(n / 2 + n / 4), int(m / 2 - m / 4): int(m / 2 + m / 4)]
                    patches = image.extract_patches_2d(mid_ima, (32, 32))
                    random_index = random.randrange(len(patches))
                    patch = patches[random_index]
                    patch = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB)
                    patch = patch / 255
                    img = np.reshape(patch, 32 * 32 * 3)

                    labels.append(index_label)
                    # path, patch_index
                    writer.writerow([subdir_class + '/' + imagen, random_index])
                    if first:
                        dataset = img
                        first = False
                    else:
                        dataset = np.vstack((dataset, img))
            index_label += 1
    labels = np.asarray(labels)
    logs_file.close()
    return dataset, labels


def load_google_colors():
    path = "/content/ClusterGAN/colors/google_colors"
    dirs = os.listdir(path)
    print(dirs)
    labels = []
    index_label = 0
    first = True
    for item in dirs:
        subdir = path + '/' + item
        print(subdir)
        if os.path.isdir(subdir):
            for image in os.listdir(subdir):
                if os.path.isfile(subdir + '/' + image):
                    bgr = cv2.imread(subdir + '/' + image)
                    # img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                    img = bgr[:, :, [2, 1, 0]]
                    img = np.reshape(img, 32 * 32 * 3)
                    img = img / 255
                    labels.append(index_label)
                    if first:
                        dataset = img
                        first = False
                    else:
                        dataset = np.vstack((dataset, img))
            index_label += 1
    labels = np.asarray(labels)
    return dataset, labels


def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return dist_2.argsort()


def closest(X, p):
    disp = X - p
    return np.argmin((disp * disp).sum(1))


def sample_Z(batch, z_dim, sampler='one_hot', num_class=10, n_cat=1, label_index=None, save_label=False):
    if sampler == 'mul_cat':
        if label_index is None:
            label_index = np.random.randint(low=0, high=num_class, size=batch)
        return np.hstack((0.10 * np.random.randn(batch, z_dim - num_class * n_cat),
                          np.tile(np.eye(num_class)[label_index], (1, n_cat))))
    elif sampler == 'one_hot':
        if label_index is None:
            label_index = np.random.randint(low=0, high=num_class, size=batch)
        if save_label:
            return np.hstack(
                (0.10 * np.random.randn(batch, z_dim - num_class), np.eye(num_class)[label_index])), label_index
        else:
            return np.hstack((0.10 * np.random.randn(batch, z_dim - num_class), np.eye(num_class)[label_index]))
    elif sampler == 'uniform':
        return np.random.uniform(-1., 1., size=[batch, z_dim])
    elif sampler == 'normal':
        return 0.15 * np.random.randn(batch, z_dim)
    elif sampler == 'mix_gauss':
        if label_index is None:
            label_index = np.random.randint(low=0, high=num_class, size=batch)
        return (0.1 * np.random.randn(batch, z_dim) + np.eye(num_class, z_dim)[label_index])


def sample_labelled_Z(batch, z_dim, sampler='one_hot', num_class=10, n_cat=1, label_index=None):
    if sampler == 'mul_cat':
        if label_index is None:
            label_index = np.random.randint(low=0, high=num_class, size=batch)
        return label_index, np.hstack((0.10 * np.random.randn(batch, z_dim - num_class * n_cat),
                                       np.tile(np.eye(num_class)[label_index], (1, n_cat))))
    elif sampler == 'one_hot':
        if label_index is None:
            label_index = np.random.randint(low=0, high=num_class, size=batch)
        return label_index, np.hstack(
            (0.10 * np.random.randn(batch, z_dim - num_class), np.eye(num_class)[label_index]))
    elif sampler == 'mix_gauss':
        if label_index is None:
            label_index = np.random.randint(low=0, high=num_class, size=batch)
        return label_index, (0.1 * np.random.randn(batch, z_dim) + np.eye(num_class, z_dim)[label_index])


def reshape_mnist(X):
    return X.reshape(X.shape[0], 28, 28, 1)


def clus_sample_Z(batch, dim_gen=20, dim_c=2, num_class=10, label_index=None):
    if label_index is None:
        label_index = np.random.randint(low=0, high=num_class, size=batch)
    batch_mat = np.zeros((batch, num_class * dim_c))
    for b in range(batch):
        batch_mat[b, label_index[b] * dim_c:(label_index[b] + 1) * dim_c] = np.random.normal(loc=1.0, scale=0.05,
                                                                                             size=(1, dim_c))
    return np.hstack((0.10 * np.random.randn(batch, dim_gen), batch_mat))


def clus_sample_labelled_Z(batch, dim_gen=20, dim_c=2, num_class=10, label_index=None):
    if label_index is None:
        label_index = np.random.randint(low=0, high=num_class, size=batch)
    batch_mat = np.zeros((batch, num_class * dim_c))
    for b in range(batch):
        batch_mat[b, label_index[b] * dim_c:(label_index[b] + 1) * dim_c] = np.random.normal(loc=1.0, scale=0.05,
                                                                                             size=(1, dim_c))
    return label_index, np.hstack((0.10 * np.random.randn(batch, dim_gen), batch_mat))


def sample_info(batch, z_dim, sampler='one_hot', num_class=10, n_cat=1, label_index=None):
    if sampler == 'one_hot':
        if label_index is None:
            label_index = np.random.randint(low=0, high=num_class, size=batch)
        return label_index, np.hstack(
            (np.random.randn(batch, z_dim - num_class), np.eye(num_class)[label_index]))
    elif sampler == 'mul_cat':
        if label_index is None:
            label_index = np.random.randint(low=0, high=num_class, size=batch)
        return label_index, np.hstack((np.random.randn(batch, z_dim - num_class * n_cat),
                                       np.tile(np.eye(num_class)[label_index], (1, n_cat))))


if __name__ == '__main__':
    load_termisk_reduced()
