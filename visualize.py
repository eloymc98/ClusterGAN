import os

import numpy as np
import matplotlib
from sklearn.manifold import TSNE

matplotlib.use('Agg')

import matplotlib.pyplot as plt


def split(x):
    assert type(x) == int
    t = int(np.floor(np.sqrt(x)))
    for a in range(t, 0, -1):
        if x % a == 0:
            return a, x / a


def grid_transform(x, size):
    a, b = split(x.shape[0])
    h, w, c = size[0], size[1], size[2]
    b = int(b)
    x = np.reshape(x, [a, b, h, w, c])
    x = np.transpose(x, [0, 2, 1, 3, 4])
    x = np.reshape(x, [a * h, b * w, c])
    if x.shape[2] == 1:
        x = np.squeeze(x, axis=2)
    return x


def grid_show(fig, x, size):
    ax = fig.add_subplot(111)
    x = grid_transform(x, size)
    if len(x.shape) > 2:
        ax.imshow(x)
    else:
        ax.imshow(x, cmap='gray')


def latent_space(latent_pts, labels_true, labels_pred, K):
    # TSNE setup
    n_samples = latent_pts.shape[0]
    perplexity = 30

    # Latent space info
    latent_dim = latent_pts.shape[1]
    n_c = K

    # Load TSNE
    if (perplexity < 0):
        tsne = TSNE(n_components=2, verbose=1, init='pca', random_state=0)
        fig_title = "PCA Initialization"
        figname = 'tsne-pca.png'
    else:
        tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=300)
        fig_title = "Perplexity = $%d$" % perplexity
        figname = 'tsne-plex%i.png' % perplexity

    # Encode real images
    enc_zn, enc_zc, enc_zc_logits = encoder(c_imgs)
    # Stack latent space encoding
    enc = np.hstack((enc_zn.cpu().detach().numpy(), enc_zc_logits.cpu().detach().numpy()))
    # enc = np.hstack((enc_zn.cpu().detach().numpy(), enc_zc.cpu().detach().numpy()))

    # Cluster with TSNE
    tsne_enc = tsne.fit_transform(enc)

    # Convert to numpy for indexing purposes
    labels = labels.cpu().data.numpy()

    # Color and marker for each true class
    colors = cm.rainbow(np.linspace(0, 1, n_c))
    markers = matplotlib.markers.MarkerStyle.filled_markers

    # Save TSNE figure to file
    fig, ax = plt.subplots(figsize=(16, 10))
    for iclass in range(0, n_c):
        # Get indices for each class
        idxs = labels == iclass
        # Scatter those points in tsne dims
        ax.scatter(tsne_enc[idxs, 0],
                   tsne_enc[idxs, 1],
                   marker=markers[iclass],
                   c=colors[iclass],
                   edgecolor=None,
                   label=r'$%i$' % iclass)

    ax.set_title(r'%s' % fig_title, fontsize=24)
    ax.set_xlabel(r'$X^{\mathrm{tSNE}}_1$', fontsize=18)
    ax.set_ylabel(r'$X^{\mathrm{tSNE}}_2$', fontsize=18)
    plt.legend(title=r'Class', loc='best', numpoints=1, fontsize=16)
    plt.tight_layout()
    fig.savefig(figname)


if __name__ == '__main__':
    from keras.datasets import cifar10
    from scipy.misc import imsave
    import pdb

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    shape = x_train[0].shape
    bx = x_train[0:64, :]
    bx = grid_transform(bx, shape)

    imsave('cifar_batch.png', bx)

    pdb.set_trace()

    print('Done !')
