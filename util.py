import numpy as np
import os
import cv2


def load_termisk_reduced():
    path = "/content/termisk_dataset"
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
            for label in classes:
                class_path = subdir + '/' + label
                count = 0
                if label in ('5', '6', '8', '10', '11'):
                    for image in os.listdir(class_path):
                        if os.path.isfile(class_path + '/' + image) and image.endswith('.png'):
                            img = cv2.imread(class_path + '/' + image, cv2.IMREAD_GRAYSCALE)
                            # img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
                            img = np.reshape(img, 96 * 96)
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
    print(classes)
    labels = []
    index_label = 0
    first = True
    for color in classes:
        subdir_class = path + '/' + color
        print(subdir_class)
        if os.path.isdir(subdir_class):
            for image in os.listdir(subdir_class):
                if os.path.isfile(subdir_class + '/' + image):
                    bgr = cv2.imread(subdir_class + '/' + image)
                    bgr = cv2.resize(bgr, (32, 32), interpolation=cv2.INTER_AREA)
                    # img = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
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


def load_cub():
    path = "/content/ClusterGAN/CUB_200_2011/CUB_200_2011/images"
    classes = os.listdir(path)
    print(classes)
    labels = []
    index_label = 0
    first = True
    for cub_class in classes:
        subdir_class = path + '/' + cub_class
        print(subdir_class)
        if os.path.isdir(subdir_class):
            for image in os.listdir(subdir_class):
                if os.path.isfile(subdir_class + '/' + image):
                    bgr = cv2.imread(subdir_class + '/' + image)
                    bgr = cv2.resize(bgr, (94, 94), interpolation=cv2.INTER_AREA)
                    img = bgr[:, :, [2, 1, 0]]
                    img = np.reshape(img, 94 * 94 * 3)
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
    l = sample_Z(10, 22, 'mul_cat', 10, 2)
    print(l)
