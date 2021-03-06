import os
import time
import dateutil.tz
import datetime
import argparse
import importlib
import tensorflow as tf
from imageio import imwrite
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from math import floor
import metric
from visualize import *
import util
import logging
import shutil
from collections import Counter

logger = logging.getLogger()

tf.set_random_seed(0)


# tf.random.set_seed(0)


def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(src=s, dst=d, symlinks=symlinks, ignore=ignore)
        else:
            shutil.copy2(src=s, dst=dst)


class clusGAN(object):
    def __init__(self, g_net, d_net, enc_net, x_sampler, z_sampler, data, model, sampler,
                 num_classes, dim_gen, n_cat, batch_size, beta_cycle_gen, beta_cycle_label):
        self.model = model
        self.data = data
        self.sampler = sampler
        self.g_net = g_net
        self.d_net = d_net
        self.enc_net = enc_net
        self.x_sampler = x_sampler
        self.z_sampler = z_sampler
        self.num_classes = num_classes
        self.dim_gen = dim_gen
        self.n_cat = n_cat
        self.batch_size = batch_size
        scale = 10.0
        self.beta_cycle_gen = beta_cycle_gen
        self.beta_cycle_label = beta_cycle_label

        self.x_dim = self.d_net.x_dim
        self.z_dim = self.g_net.z_dim

        self.x = tf.placeholder(tf.float32, [None, self.x_dim], name='x')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')

        self.z_gen = self.z[:, 0:self.dim_gen]
        self.z_hot = self.z[:, self.dim_gen:]

        # Generate x_ from generator
        self.x_ = self.g_net(self.z)
        # Encode generated images
        self.z_enc_gen, self.z_enc_label, self.z_enc_logits = self.enc_net(self.x_, reuse=False)
        # Encode inferred images
        self.z_infer_gen, self.z_infer_label, self.z_infer_logits = self.enc_net(self.x)

        self.d = self.d_net(self.x, reuse=False)
        self.d_ = self.d_net(self.x_)

        self.g_loss = tf.reduce_mean(self.d_) + \
                      self.beta_cycle_gen * tf.reduce_mean(tf.square(self.z_gen - self.z_enc_gen)) + \
                      self.beta_cycle_label * tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.z_enc_logits, labels=self.z_hot))

        self.g_loss_reduce_mean = tf.reduce_mean(self.d_)
        self.g_loss_beta_cycle_gen = self.beta_cycle_gen * tf.reduce_mean(tf.square(self.z_gen - self.z_enc_gen))
        self.g_loss_beta_cycle_label = self.beta_cycle_label * tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.z_enc_logits, labels=self.z_hot))

        self.output_d_ = self.d_

        self.d_loss = tf.reduce_mean(self.d) - tf.reduce_mean(self.d_)

        epsilon = tf.random_uniform([], 0.0, 1.0)
        print(f'Self.x.shape = {self.x.shape}, Self x_ = {self.x_.shape}')
        x_hat = epsilon * self.x + (1 - epsilon) * self.x_
        d_hat = self.d_net(x_hat)

        ddx = tf.gradients(d_hat, x_hat)[0]
        ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=1))
        ddx = tf.reduce_mean(tf.square(ddx - 1.0) * scale)

        self.d_loss = self.d_loss + ddx

        self.d_adam = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9) \
            .minimize(self.d_loss, var_list=self.d_net.vars)
        self.g_adam = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9) \
            .minimize(self.g_loss, var_list=[self.g_net.vars, self.enc_net.vars])

        # Reconstruction Nodes
        self.recon_loss = tf.reduce_mean(tf.abs(self.x - self.x_), 1)
        self.compute_grad = tf.gradients(self.recon_loss, self.z)

        self.saver = tf.train.Saver()

        run_config = tf.ConfigProto()
        run_config.gpu_options.per_process_gpu_memory_fraction = 1.0
        run_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=run_config)

    def train(self, num_batches=500000):

        now = datetime.datetime.now(dateutil.tz.tzlocal())
        epoch = 0
        timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

        batch_size = self.batch_size
        plt.ion()
        if args.from_checkpoint == 'False':
            self.sess.run(tf.global_variables_initializer())
            print('Initialized global variables!!!')
        start_time = time.time()
        print(
            'Training {} on {}, sampler = {}, z = {} dimension, beta_n = {}, beta_c = {}'.
                format(self.model, self.data, self.sampler, self.z_dim, self.beta_cycle_gen, self.beta_cycle_label))

        im_save_dir = 'logs/{}/{}/{}_z{}_cyc{}_gen{}'.format(self.data, self.model, self.sampler, self.z_dim,
                                                             self.beta_cycle_label, self.beta_cycle_gen)
        if not os.path.exists(im_save_dir):
            os.makedirs(im_save_dir)

        if args.data == 'cifar':
            train_size = 40000
        elif args.data == 'colors':
            train_size = 880
        elif args.data in ('mnist', 'fashion'):
            train_size = 60000
        elif args.data == 'colors_new':
            train_size = 2747
        elif args.data == 'synthetic_colors':
            train_size = 1330

        for t in range(0, num_batches):
            d_iters = 5

            for _ in range(0, d_iters):
                # We optimize discriminator loss for 5 iterations
                bx = self.x_sampler.train(batch_size)
                bz = self.z_sampler(batch_size, self.z_dim, self.sampler, self.num_classes, self.n_cat)

                self.sess.run(self.d_adam, feed_dict={self.x: bx, self.z: bz})
            # Then we only optimize generator loss for 1 iteration
            bz = self.z_sampler(batch_size, self.z_dim, self.sampler, self.num_classes, self.n_cat)
            self.sess.run(self.g_adam, feed_dict={self.z: bz})
            # if (t + 1) % floor(train_size / (batch_size * d_iters)) == 0:
            #     print(f'Epoch {epoch}')
            #     epoch += 1

            if (t + 1) % 100 == 0:
                # Every 100 iter, print d and g losses
                bx = self.x_sampler.train(batch_size)
                bz = self.z_sampler(batch_size, self.z_dim, self.sampler, self.num_classes, self.n_cat)

                d_loss = self.sess.run(
                    self.d_loss, feed_dict={self.x: bx, self.z: bz}
                )
                g_loss = self.sess.run(
                    self.g_loss, feed_dict={self.z: bz}
                )
                g_loss_reduce_mean = self.sess.run(self.g_loss_reduce_mean, feed_dict={self.z: bz})
                g_loss_beta_cycle_gen = self.sess.run(self.g_loss_beta_cycle_gen, feed_dict={self.z: bz})
                g_loss_beta_cycle_label = self.sess.run(self.g_loss_beta_cycle_label, feed_dict={self.z: bz})
                out_d_ = self.sess.run(self.output_d_, feed_dict={self.z: bz})

                print(
                    'Iter [%8d] Time [%5.4f] d_loss [%.4f] g_loss [%.4f] (reduce_mean [%.4f], b_n [%.4f], b_c [%.4f])' %
                    (t + 1, time.time() - start_time, d_loss, g_loss, g_loss_reduce_mean, g_loss_beta_cycle_gen,
                     g_loss_beta_cycle_label))
                print(f'd_: {out_d_[0]}')

            if (t + 1) % 5000 == 0:
                # Every 5000 iter, save an image of a batch of x_
                bz, labels_indexx = self.z_sampler(batch_size, self.z_dim, self.sampler, self.num_classes, self.n_cat,
                                                   save_label=True)
                bx = self.sess.run(self.x_, feed_dict={self.z: bz})
                bx = xs.data2img(bx)
                # if 'colors' in args.data:
                #     import cv2
                #     print(f'CIE-LAB! Bx: {bx.shape[0]}')
                #     for i in range(bx.shape[0]):
                #         bx[i] = cv2.cvtColor((bx[i] * 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
                #         bx[i] = bx[i] / 255

                bx = grid_transform(bx, xs.shape)

                imwrite('logs/{}/{}/{}_z{}_cyc{}_gen{}/{}.png'.format(self.data, self.model, self.sampler,
                                                                      self.z_dim, self.beta_cycle_label,
                                                                      self.beta_cycle_gen, (t + 1) / 100), bx)
                with open('logs/{}/{}/{}_z{}_cyc{}_gen{}/{}.txt'.format(self.data, self.model, self.sampler,
                                                                        self.z_dim, self.beta_cycle_label,
                                                                        self.beta_cycle_gen, (t + 1) / 100), 'w') as f:
                    for i in labels_indexx:
                        f.write(str(i))
                        f.write(', ')

            if (t + 1) % 5000 == 0:
                self.save(timestamp + '_' + str(t + 1), t=t)

        self.recon_enc(timestamp, val=True)
        self.save(timestamp)

    def save(self, timestamp, t=None):

        checkpoint_dir = 'checkpoint_dir/{}/{}_{}_{}_z{}_cyc{}_gen{}'.format(self.data, timestamp, self.model,
                                                                             self.sampler,
                                                                             self.z_dim, self.beta_cycle_label,
                                                                             self.beta_cycle_gen)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, 'model.ckpt'))

        date = datetime.datetime.utcnow().strftime("%Y-%m-%d-%H-M-%S")
        # if not os.path.exists('/content/gdrive/My Drive/ClusterGAN/checkpoints'):
        #     os.makedirs('/content/gdrive/My Drive/ClusterGAN/checkpoints')
        #     os.makedirs(f'/content/gdrive/My Drive/ClusterGAN/checkpoints/{self.data}')
        if self.data == 'fashion':
            copytree(checkpoint_dir,
                     f'/content/gdrive/My Drive/ClusterGAN/checkpoints/fashion_{self.num_classes}')
            copytree('logs/{}/{}/{}_z{}_cyc{}_gen{}'.format(self.data, self.model, self.sampler,
                                                            self.z_dim, self.beta_cycle_label,
                                                            self.beta_cycle_gen),
                     f'/content/gdrive/My Drive/ClusterGAN/checkpoints/fashion_{self.num_classes}')
        else:
            copytree(checkpoint_dir,
                     f'/content/gdrive/My Drive/ClusterGAN/checkpoints/{self.data}')
            copytree('logs/{}/{}/{}_z{}_cyc{}_gen{}'.format(self.data, self.model, self.sampler,
                                                            self.z_dim, self.beta_cycle_label,
                                                            self.beta_cycle_gen),
                     f'/content/gdrive/My Drive/ClusterGAN/checkpoints/{self.data}')

    def load(self, pre_trained=False, timestamp=''):

        if pre_trained == True:
            print('Loading Pre-trained Model...')
            checkpoint_dir = 'pre_trained_models/{}/{}_{}_z{}_cyc{}_gen{}'.format(self.data, self.model, self.sampler,
                                                                                  self.z_dim, self.beta_cycle_label,
                                                                                  self.beta_cycle_gen)
            if self.data in ('termisk', 'cifar', 'mnist', 'fashion', 'colors', 'colors_new', 'synthetic_colors'):
                checkpoint_dir = 'pre_trained_models/{}'.format(self.data)
        else:
            if timestamp == '':
                print('Best Timestamp not provided. Abort !')
                checkpoint_dir = ''
            else:
                checkpoint_dir = 'checkpoint_dir/{}/{}_{}_{}_z{}_cyc{}_gen{}'.format(self.data, timestamp, self.model,
                                                                                     self.sampler,
                                                                                     self.z_dim, self.beta_cycle_label,
                                                                                     self.beta_cycle_gen)

        self.saver.restore(self.sess, os.path.join(checkpoint_dir, 'model.ckpt'))

        print('Restored model weights.')

    def _gen_samples(self, num_images):

        batch_size = self.batch_size
        bz = self.z_sampler(batch_size, self.z_dim, self.sampler, self.num_classes, self.n_cat)
        fake_im = self.sess.run(self.x_, feed_dict={self.z: bz})
        for t in range(num_images // batch_size):
            bz = self.z_sampler(batch_size, self.z_dim, self.sampler, self.num_classes, self.n_cat)
            im = self.sess.run(self.x_, feed_dict={self.z: bz})
            fake_im = np.vstack((fake_im, im))

        print(' Generated {} images .'.format(fake_im.shape[0]))
        np.save('./Image_samples/{}/{}_{}_K_{}_gen_images.npy'.format(self.data, self.model, self.sampler,
                                                                      self.num_classes), fake_im)

    def gen_from_all_modes(self):

        if self.sampler == 'one_hot':
            if self.num_classes in (16, 12, 14):
                batch_size = 1008
            elif self.num_classes == 5:
                batch_size = 1000
            elif self.num_classes in (11, 13):
                batch_size = 1001
            elif self.num_classes == 9:
                batch_size = 1008
            elif self.num_classes == 22:
                batch_size = 1012
            else:
                batch_size = 1000
            label_index = np.tile(np.arange(self.num_classes), int(np.ceil(batch_size * 1.0 / self.num_classes)))
            print(f'Label index {label_index.shape}')

            bz = self.z_sampler(batch_size, self.z_dim, self.sampler, num_class=self.num_classes,
                                n_cat=self.n_cat, label_index=label_index)
            bx = self.sess.run(self.x_, feed_dict={self.z: bz})
            for m in range(self.num_classes):
                print('Generating samples from mode {} ...'.format(m))
                mode_index = np.where(label_index == m)[0]
                mode_bx = bx[mode_index, :]
                mode_bx = xs.data2img(mode_bx)
                if 'colors' in args.data:
                    import cv2
                    print(f'CIE-LAB! Bx: {mode_bx.shape[0]}')
                    for i in range(mode_bx.shape[0]):
                        mode_bx[i] = cv2.cvtColor((mode_bx[i] * 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
                        mode_bx[i] = mode_bx[i] / 255
                mode_bx = grid_transform(mode_bx, xs.shape)

                if not os.path.exists('logs/'):
                    os.makedirs('logs/')

                imwrite('logs/{}_mode{}_samples.png'.format(self.data, m), mode_bx)

    def recon_enc(self, timestamp, val=True, mapping=None):

        if val:
            data_recon, label_recon = self.x_sampler.validation()
        elif 'colors' in self.data:
            data_recon, label_recon, ima_names = self.x_sampler.test(index=True)
            # if 'colors' in args.data:
            #     import cv2
            #     data_recon = xs.data2img(data_recon)
            #     print(f'CIE-LAB! Bx: {data_recon.shape[0]}')
            #     for i in range(data_recon.shape[0]):
            #         data_recon[i] = cv2.cvtColor((data_recon[i] * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
            #         data_recon[i] = data_recon[i] / 255
            #     data_recon = np.reshape(data_recon, [data_recon.shape[0]] + [32 * 32 * 3])
            # data_recon, label_recon = self.x_sampler.load_all()
        else:
            data_recon, label_recon = self.x_sampler.test()

        num_pts_to_plot = data_recon.shape[0]  # num of images
        recon_batch_size = self.batch_size
        latent = np.zeros(shape=(num_pts_to_plot, self.z_dim))
        labelsss = np.zeros(shape=(num_pts_to_plot))
        print('Data Shape = {}, Labels Shape = {}'.format(data_recon.shape, label_recon.shape))
        for b in range(int(np.ceil(num_pts_to_plot * 1.0 / recon_batch_size))):
            if (b + 1) * recon_batch_size > num_pts_to_plot:
                pt_indx = np.arange(b * recon_batch_size, num_pts_to_plot)
            else:
                pt_indx = np.arange(b * recon_batch_size, (b + 1) * recon_batch_size)
            xtrue = data_recon[pt_indx, :]

            zhats_gen, zhats_label = self.sess.run([self.z_infer_gen, self.z_infer_label], feed_dict={self.x: xtrue})

            labelsss[pt_indx] = np.argmax(zhats_label, axis=1)
            if b == 1:
                print(f'labels {labelsss}')
            latent[pt_indx, :] = np.concatenate((zhats_gen, zhats_label), axis=1)

        if 'colors' in self.data:
            final_labels_predicted = []
            final_true_labels = []
            past_true_label = label_recon[0]
            past_ima_index = ima_names[0]
            ima_predictions = []
            for i in range(num_pts_to_plot):
                label_pred = labelsss[i]
                true_label = label_recon[i]
                ima_index = ima_names[i]
                if ima_index != past_ima_index:
                    final_labels_predicted.append(self.most_frequent(ima_predictions))
                    final_true_labels.append(past_true_label)
                    ima_predictions = []
                ima_predictions.append(label_pred)
                past_true_label = true_label
                past_ima_index = ima_index

            label_recon = np.asarray(final_true_labels)
            labelsss = np.asarray(final_labels_predicted)

        if self.beta_cycle_gen == 0:
            km = self._eval_cluster(latent[:, self.dim_gen:], label_recon, timestamp, val, labelsss, mapping)
        else:
            km = self._eval_cluster(latent, label_recon, timestamp, val, labelsss, mapping)
        return km, latent

    def _eval_cluster(self, latent_rep, labels_true, timestamp, val, labels_predicted, mapping):

        if self.data == 'fashion' and self.num_classes == 5:
            map_labels = {0: 0, 1: 1, 2: 2, 3: 0, 4: 2, 5: 3, 6: 2, 7: 3, 8: 4, 9: 3}
            labels_true = np.array([map_labels[i] for i in labels_true])
        elif self.data == 'colors':
            # si ha caido en el cluster 0 es para el color 10, cluster 1 para el color 8, ...
            map_labels = {}
            # Row_ind: [0  1  2  3  4  5  6  7  8  9 10]
            # Col_ind: [10  8  5  7  6  4  1  0  9  2  3]

        km = KMeans(n_clusters=max(self.num_classes, len(np.unique(labels_true))), random_state=0, verbose=1).fit(
            latent_rep)
        labels_pred = km.labels_
        print(labels_true.shape)
        print(labels_pred.shape)
        print(labels_predicted.shape)
        # type labels_predicted np.ndarray
        # purity = metric.compute_purity(labels_pred, labels_true)
        # ari = adjusted_rand_score(labels_true, labels_pred)
        # nmi = normalized_mutual_info_score(labels_true, labels_pred)

        random_labels = np.random.randint(low=0, high=self.num_classes - 1, size=len(labels_predicted))

        purity2 = metric.compute_purity(labels_predicted, labels_true)
        ari2 = adjusted_rand_score(labels_true, labels_predicted)
        nmi2 = normalized_mutual_info_score(labels_true, labels_predicted)

        purity_random = metric.compute_purity(random_labels, labels_true)
        ari_random = adjusted_rand_score(labels_true, random_labels)
        nmi_random = normalized_mutual_info_score(labels_true, random_labels)

        if mapping:
            acc = metric.compute_purity_assigned(labels_predicted, labels_true, mapping)

            print(f'ACCURACYYYYY! {acc}')

        if val:
            data_split = 'Validation'
        else:
            data_split = 'Test'
            # data_split = 'All'

        print('Data = {}, Model = {}, sampler = {}, z_dim = {}, beta_label = {}, beta_gen = {} '
              .format(self.data, self.model, self.sampler, self.z_dim, self.beta_cycle_label, self.beta_cycle_gen))
        print(
            ' #Points = {}, K = {}, Purity = {},  NMI = {}, ARI = {}, Latent space shape = {}, P2={}, nmi2={}, ari2={} '
                .format(latent_rep.shape[0], self.num_classes, purity2, nmi2, ari2, latent_rep.shape, purity2, nmi2,
                        ari2))
        print(' RANDOM!!!!!!!  Purity = {},  NMI = {}, ARI = {}'.format(purity_random, nmi_random, ari_random))

        print('Latent')
        # Esto es un punto en el espacio n-dim.
        print(latent_rep[0])

        # latent_space(latent_rep, labels_true, labels_pred, self.num_classes)

        if not os.path.exists('logs'):
            os.makedirs('logs')

        with open('logs/Res_{}_{}.txt'.format(self.data, self.model), 'a+') as f:
            f.write(
                '{}, {} : K = {}, z_dim = {}, beta_label = {}, beta_gen = {}, sampler = {}, Purity = {}, NMI = {}, ARI = {}\n'
                    .format(timestamp, data_split, self.num_classes, self.z_dim, self.beta_cycle_label,
                            self.beta_cycle_gen,
                            self.sampler, purity2, nmi2, ari2))
            f.flush()

        return km

    def label_img(self, path, km, latent):
        from pathlib import Path
        import cv2
        query = Path(path)

        if query.is_file():

            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            res = cv2.resize(img, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
            res = res.flatten() / 255
            latent_pt = np.zeros(shape=(1, self.z_dim))
            x = np.zeros(shape=(1, res.shape[0]))
            x[0, :] = res
            zhats_gen, zhats_label = self.sess.run([self.z_infer_gen, self.z_infer_label], feed_dict={self.x: x})

            latent_pt[0, :] = np.concatenate((zhats_gen, zhats_label), axis=1)

            # clusters = km.cluster_centers_  # (n_clusters, features)
            # labels = km.labels_
            # index = util.closest(latent, latent_pt[0])
            # print(f'latent idx {latent[index]}')
            # print(latent_pt[0])
            # print(f'index {index}')
            # print(f'label {labels[index]}')
            print(zhats_label)
            label = np.argmax(zhats_label, axis=1)
            print(label)

        elif query.is_dir():
            return None

    def euclidean_distance(row1, row2):
        from math import sqrt
        distance = 0.0
        for i in range(len(row1) - 1):
            distance += (row1[i] - row2[i]) ** 2
        return sqrt(distance)

    def label_test_images(self):
        from sklearn.metrics import accuracy_score, confusion_matrix
        import seaborn as sn
        import pandas as pd
        from sklearn.neighbors import KNeighborsClassifier

        data, true_labels = self.x_sampler.test()
        test_zhats_gen, test_zhats_label, zhats_logits = self.sess.run(
            [self.z_infer_gen, self.z_infer_label, self.z_infer_logits], feed_dict={self.x: data})

        cluster_to_label_mapping = {0: 1, 1: 7, 2: 2, 3: 4, 4: 0, 5: 3, 6: 8, 7: 6, 8: 5, 9: 9}

        pred_labels_argmax = np.argmax(test_zhats_label, axis=1)
        labels_pred_argmax = np.array([cluster_to_label_mapping[c] for c in pred_labels_argmax])
        acc_argmax = accuracy_score(true_labels, labels_pred_argmax)
        print(f'ACCURACY with argmax: {acc_argmax}')
        cm = confusion_matrix(true_labels, labels_pred_argmax)
        df_cm = pd.DataFrame(cm, range(self.num_classes), range(self.num_classes))
        print(df_cm.head())
        sn.set(font_scale=1.4)  # for label size
        sn.heatmap(df_cm)  # font size
        plt.savefig("confusion_matrix_argmax.png")

        training_data, training_labels = self.x_sampler.load_all_train()
        zhats_gen, zhats_label = self.sess.run([self.z_infer_gen, self.z_infer_label], feed_dict={self.x: training_data})

        #crear un dataframe con training data, 1 column por dim + col de label
        df_training = pd.DataFrame({'v0': zhats_gen[:, 0], 'v1': zhats_gen[:, 1], 'v2': zhats_gen[:, 2], 'v3': zhats_gen[:, 3],
                                    'v4': zhats_gen[:, 4], 'v5': zhats_gen[:, 5], 'v6': zhats_gen[:, 6], 'v7': zhats_gen[:, 7],
                                    'v8': zhats_gen[:, 8], 'v9': zhats_gen[:, 9], 'v10': zhats_gen[:, 10], 'v11': zhats_gen[:, 11],
                                    'v12': zhats_gen[:, 12], 'v13': zhats_gen[:, 13], 'v14': zhats_gen[:, 14], 'v15': zhats_gen[:, 15],
                                    'v16': zhats_gen[:, 16], 'v17': zhats_gen[:, 17], 'v18': zhats_gen[:, 18], 'v19': zhats_gen[:, 19],
                                    'v20': zhats_gen[:, 20], 'v21': zhats_gen[:, 21], 'v22': zhats_gen[:, 22], 'v23': zhats_gen[:, 23],
                                    'v24': zhats_gen[:, 24], 'v25': zhats_gen[:, 25], 'v26': zhats_gen[:, 26], 'v27': zhats_gen[:, 27],
                                    'v28': zhats_gen[:, 28], 'v29': zhats_gen[:, 29], 'v30': zhats_label[:, 0], 'v31': zhats_label[:, 1],
                                    'v32': zhats_label[:, 2], 'v33': zhats_label[:, 3], 'v34': zhats_label[:, 4], 'v35': zhats_label[:, 5],
                                    'v36': zhats_label[:, 6], 'v37': zhats_label[:, 7], 'v38': zhats_label[:, 8], 'v39': zhats_label[:, 9],
                                    'label': np.argmax(zhats_label, axis=1)
                                    })
        
        df_test = pd.DataFrame({'v0': test_zhats_gen[:, 0], 'v1': test_zhats_gen[:, 1], 'v2': test_zhats_gen[:, 2], 'v3': test_zhats_gen[:, 3],
                                    'v4': test_zhats_gen[:, 4], 'v5': test_zhats_gen[:, 5], 'v6': test_zhats_gen[:, 6], 'v7': test_zhats_gen[:, 7],
                                    'v8': test_zhats_gen[:, 8], 'v9': test_zhats_gen[:, 9], 'v10': test_zhats_gen[:, 10], 'v11': test_zhats_gen[:, 11],
                                    'v12': test_zhats_gen[:, 12], 'v13': test_zhats_gen[:, 13], 'v14': test_zhats_gen[:, 14], 'v15': test_zhats_gen[:, 15],
                                    'v16': test_zhats_gen[:, 16], 'v17': test_zhats_gen[:, 17], 'v18': test_zhats_gen[:, 18], 'v19': test_zhats_gen[:, 19],
                                    'v20': test_zhats_gen[:, 20], 'v21': test_zhats_gen[:, 21], 'v22': test_zhats_gen[:, 22], 'v23': test_zhats_gen[:, 23],
                                    'v24': test_zhats_gen[:, 24], 'v25': test_zhats_gen[:, 25], 'v26': test_zhats_gen[:, 26], 'v27': test_zhats_gen[:, 27],
                                    'v28': test_zhats_gen[:, 28], 'v29': test_zhats_gen[:, 29], 'v30': test_zhats_label[:, 0], 'v31': test_zhats_label[:, 1],
                                    'v32': test_zhats_label[:, 2], 'v33': test_zhats_label[:, 3], 'v34': test_zhats_label[:, 4], 'v35': test_zhats_label[:, 5],
                                    'v36': test_zhats_label[:, 6], 'v37': test_zhats_label[:, 7], 'v38': test_zhats_label[:, 8], 'v39': test_zhats_label[:, 9]
                                    })
        print(df_training.head())
        neigh = KNeighborsClassifier(n_neighbors=5)
        neigh.fit(df_training.iloc[:, 0:40], df_training['label'])
        knn_predictions = neigh.predict(df_test.iloc[:, 0:40])
        knn_predictions = np.array([cluster_to_label_mapping[c] for c in knn_predictions])
        print(knn_predictions.shape)
        acc_knn = accuracy_score(true_labels, knn_predictions)
        print(f'ACCURACY with knn: {acc_knn}')
        cm2 = confusion_matrix(true_labels, knn_predictions)
        df_cm2 = pd.DataFrame(cm2, range(self.num_classes), range(self.num_classes))
        print(df_cm2.head())
        sn.set(font_scale=1.4)  # for label size
        sn.heatmap(df_cm2)  # font size
        plt.savefig("confusion_matrix_knn2.png")



    def encoder_to_gen(self, bx):
        # bx, bx_labels = self.x_sampler.test()
        # solo coger unas cuantas, no las 2758 imagenes
        # bx = bx[0:100]
        zhats_gen, zhats_label, zhats_logits = self.sess.run(
            [self.z_infer_gen, self.z_infer_label, self.z_infer_logits], feed_dict={self.x: bx})
        print(f'Shape: {zhats_gen.shape}, z_gen:  {zhats_gen}')
        print(f'Shape: {zhats_label.shape}, z_label:  {np.around(zhats_label, decimals=0)}')
        print(f'Shape: {zhats_logits.shape}, z_logits:  {zhats_logits}')
        bz = np.hstack((zhats_gen, np.around(zhats_label, decimals=0)))
        print(np.argmax(zhats_label, axis=1))
        print(f'z.shape: {bz.shape}')
        bx_ = self.sess.run(self.x_, feed_dict={self.z: bz})
        print(f'bx_: {bx_.shape}')
        bx_ = xs.data2img(bx_)
        if 'colors' in args.data:
            import cv2
            print(f'CIE-LAB! Bx: {bx_.shape[0]}')
            for i in range(bx.shape[0]):
                bx_[i] = cv2.cvtColor((bx_[i] * 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
                bx_[i] = bx_[i] / 255
        bx_ = grid_transform(bx_, xs.shape)
        num = np.random.randint(0, 1000)
        imwrite(f'generated{num}.png', bx_)
        bx = xs.data2img(bx)
        bx = grid_transform(bx, xs.shape)
        imwrite(f'inferred{num}.png', bx)

    def co_matrix(self):

        import pandas as pd
        import seaborn as sn

        data_recon, label_recon = self.x_sampler.test()
        # data_recon, label_recon = self.x_sampler.load_all()
        if self.data == "fashion" and self.num_classes == 10:
            label_recon_labels = {0: 't-shirt/top', 1: 'trouser', 2: 'pullover', 3: 'dress', 4: 'coat', 5: 'sandal',
                                  6: 'shirt', 7: 'sneaker', 8: 'bag', 9: 'ankle boot'}
        elif self.data == "fashion" and self.num_classes == 5:
            label_recon_labels = {0: 't-shirt/top,dress', 1: 'trouser', 2: 'pullover,coat,shirt',
                                  3: 't-shirt/top,dress',
                                  4: 'pullover,coat,shirt', 5: 'sandal,sneaker,boot', 6: 'pullover,coat,shirt',
                                  7: 'sandal,sneaker,boot', 8: 'bag', 9: 'sandal,sneaker,boot'}
        elif self.data == "cifar":
            label_recon_labels = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog',
                                  6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
        # mode_labels = {0: ['yellow', 'orange'], 1: ['green', 'brown'], 2: ['pink', 'purple' 'blue'], 3: ['grey'],
        #                4: ['black'],
        #                5: ['white'], 6: ['black'], 7: ['red', 'orange'], 8: ['grey', 'pink'], 9: ['blue', 'brown'],
        #                10: ['white']}

        true_labels_mapped = []
        for item in list(label_recon):
            true_labels_mapped.append(label_recon_labels[item])

        num_pts_to_plot = data_recon.shape[0]  # num of images
        recon_batch_size = self.batch_size

        labels_predicted = np.zeros(shape=(num_pts_to_plot))
        # labels_predicted_mapped = []
        for b in range(int(np.ceil(num_pts_to_plot * 1.0 / recon_batch_size))):
            if (b + 1) * recon_batch_size > num_pts_to_plot:
                pt_indx = np.arange(b * recon_batch_size, num_pts_to_plot)
            else:
                pt_indx = np.arange(b * recon_batch_size, (b + 1) * recon_batch_size)
            xtrue = data_recon[pt_indx, :]

            zhats_gen, zhats_label = self.sess.run([self.z_infer_gen, self.z_infer_label], feed_dict={self.x: xtrue})

            labels_predicted[pt_indx] = np.argmax(zhats_label, axis=1)
            # print(np.argmax(zhats_label, axis=1))
            # x = np.argmax(zhats_label, axis=1)
            # for value in list(x):
            #     labels_predicted_mapped.append(mode_labels[value])

        # hacer dataframe con columnas y_real, y_pred
        df = pd.DataFrame(columns=['label', 'cluster'])
        df['label'] = true_labels_mapped
        df['cluster'] = labels_predicted.astype(np.uint8)
        print(df.head())
        df2 = df
        co_mat = pd.crosstab(df.label, df.cluster, normalize='index')
        sn.set(font_scale=2)
        # plt.tight_layout()
        # plt.figure(figsize=(14.4, 10.8))
        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_axes([0.4, 0.2, 0.5, 0.6])
        sn.heatmap(co_mat, ax=ax1)
        plt.savefig(f'{self.data}_{self.num_classes}_matrix.png')

        co_mat2 = pd.crosstab(df2.label, df2.cluster, normalize='all')
        sn.set(font_scale=2)
        # plt.tight_layout()
        # plt.figure(figsize=(14.4, 10.8))
        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_axes([0.4, 0.2, 0.5, 0.6])
        sn.heatmap(co_mat2, ax=ax1)
        plt.savefig(f'{self.data}_{self.num_classes}_matrix_all.png')
        # tengo labels reales y label generadas y el mapeo correspondiente
        # from sklearn.metrics import confusion_matrix
        # y_pred = []
        # for i in range(len(labels_predicted)):
        #     y_pred.append(labels_predicted[i])
        #
        # labels = ['black', 'blue', 'brown', 'green', 'grey', 'orange', 'pink', 'purple', 'red',
        #           'white',
        #           'yellow']
        # cm = confusion_matrix(true_labels_mapped, y_pred)
        # print(cm)
        # import seaborn as sn
        # import pandas as pd
        # df_cm = pd.DataFrame(cm, range(11), range(11))
        # print(df_cm.head())
        # sn.set(font_scale=1.4)  # for label size
        # sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size
        # plt.savefig("cm.png")

    def most_frequent(self, x):
        occurence_count = Counter(x)
        return occurence_count.most_common(1)[0][0]

    def assignment_problem(self):
        import pandas as pd
        from scipy.optimize import linear_sum_assignment

        data_recon, label_recon = self.x_sampler.load_all_train()
        num_pts_to_plot = data_recon.shape[0]  # num of images
        recon_batch_size = self.batch_size

        labels_predicted = np.zeros(shape=(num_pts_to_plot))
        zhats_gen, zhats_label = self.sess.run([self.z_infer_gen, self.z_infer_label], feed_dict={self.x: data_recon})
        labels_predicted = np.argmax(zhats_label, axis=1)
        print(f'Labels predicted co-mat shape:!!!!!! {labels_predicted.shape}')

        df = pd.DataFrame(columns=['label', 'cluster'])
        df['label'] = label_recon
        df['cluster'] = labels_predicted
        print(df.head())

        co_mat = pd.crosstab(df.label, df.cluster, normalize='all')
        print(co_mat.head())
        class_ind, cluster_ind = linear_sum_assignment(co_mat.values)
        print(f'Row_ind: {class_ind}')
        print(f'Col_ind: {cluster_ind}')
        mapping = dict(zip(cluster_ind, class_ind))
        print(mapping)

        # Row_ind: [0  1  2  3  4  5  6  7  8  9 10]
        # Col_ind: [10  8 11  1  2  3  0 12  4  9  5]
        return mapping

    def colors_confusion_matrix(self):

        import pandas as pd
        import seaborn as sn

        data_recon, label_recon, ima_names = self.x_sampler.test(index=True)

        if 'colors' in args.data:
            import cv2
            data_recon = xs.data2img(data_recon)
            print(f'CIE-LAB! Bx: {data_recon.shape[0]}')
            for i in range(data_recon.shape[0]):
                data_recon[i] = cv2.cvtColor((data_recon[i] * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
                data_recon[i] = data_recon[i] / 255
            data_recon = np.reshape(data_recon, [data_recon.shape[0]] + [32 * 32 * 3])

        # data_recon, label_recon = self.x_sampler.load_all()
        label_recon_labels = {0: 'black', 1: 'blue', 2: 'brown', 3: 'green', 4: 'grey', 5: 'orange', 6: 'pink',
                              7: 'purple', 8: 'red', 9: 'white', 10: 'yellow'}
        # mode_labels = {0: ['yellow', 'orange'], 1: ['green', 'brown'], 2: ['pink', 'purple' 'blue'], 3: ['grey'],
        #                4: ['black'],
        #                5: ['white'], 6: ['black'], 7: ['red', 'orange'], 8: ['grey', 'pink'], 9: ['blue', 'brown'],
        #                10: ['white']}

        true_labels_mapped = []
        for item in list(label_recon):
            true_labels_mapped.append(label_recon_labels[item])

        num_pts_to_plot = data_recon.shape[0]  # num of images
        recon_batch_size = self.batch_size

        labels_predicted = np.zeros(shape=(num_pts_to_plot))
        zhats_gen, zhats_label = self.sess.run([self.z_infer_gen, self.z_infer_label], feed_dict={self.x: data_recon})
        labels_predicted = np.argmax(zhats_label, axis=1)
        print(f'Labels predicted co-mat shape:!!!!!! {labels_predicted.shape}')

        final_labels_predicted = []
        final_true_labels = []
        past_true_label = true_labels_mapped[0]
        print(ima_names)
        print(ima_names.shape)
        past_ima_index = ima_names[0]
        ima_predictions = []
        for i in range(num_pts_to_plot):
            label_pred = labels_predicted[i]
            true_label = true_labels_mapped[i]
            ima_index = ima_names[i]
            if ima_index != past_ima_index:
                final_labels_predicted.append(self.most_frequent(ima_predictions))
                final_true_labels.append(past_true_label)
                ima_predictions = []
            ima_predictions.append(label_pred)
            past_true_label = true_label
            past_ima_index = ima_index
        print(f'Labels predicted finaaal len:!!!!!! {len(final_labels_predicted)}')

        # labels_predicted_mapped = []
        # for b in range(int(np.ceil(num_pts_to_plot * 1.0 / recon_batch_size))):
        #     if (b + 1) * recon_batch_size > num_pts_to_plot:
        #         pt_indx = np.arange(b * recon_batch_size, num_pts_to_plot)
        #     else:
        #         pt_indx = np.arange(b * recon_batch_size, (b + 1) * recon_batch_size)
        #     xtrue = data_recon[pt_indx, :]
        #
        #     zhats_gen, zhats_label = self.sess.run([self.z_infer_gen, self.z_infer_label], feed_dict={self.x: xtrue})
        #
        #     labels_predicted[pt_indx] = np.argmax(zhats_label, axis=1)
        # print(np.argmax(zhats_label, axis=1))
        # x = np.argmax(zhats_label, axis=1)
        # for value in list(x):
        #     labels_predicted_mapped.append(mode_labels[value])

        # hacer dataframe con columnas y_real, y_pred
        df = pd.DataFrame(columns=['label', 'cluster'])
        df['label'] = final_true_labels
        df['cluster'] = final_labels_predicted
        print(df.head())

        co_mat = pd.crosstab(df.label, df.cluster, normalize='all')
        sn.set(font_scale=1.4)
        sn.heatmap(co_mat)
        plt.savefig('co-ocurrence.png')

        from scipy.optimize import linear_sum_assignment
        print(co_mat.head())
        row_ind, col_ind = linear_sum_assignment(co_mat.values)
        print(f'Row_ind: {row_ind}')
        print(f'Col_ind: {col_ind}')
        # tengo labels reales y label generadas y el mapeo correspondiente
        # from sklearn.metrics import confusion_matrix
        # y_pred = []
        # for i in range(len(labels_predicted)):
        #     y_pred.append(labels_predicted[i])
        #
        # labels = ['black', 'blue', 'brown', 'green', 'grey', 'orange', 'pink', 'purple', 'red',
        #           'white',
        #           'yellow']
        # cm = confusion_matrix(true_labels_mapped, y_pred)
        # print(cm)
        # import seaborn as sn
        # import pandas as pd
        # df_cm = pd.DataFrame(cm, range(11), range(11))
        # print(df_cm.head())
        # sn.set(font_scale=1.4)  # for label size
        # sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size
        # plt.savefig("cm.png")

    def mnist_confusion_matrix(self):

        import pandas as pd
        import seaborn as sn

        data_recon, label_recon = self.x_sampler.test()

        num_pts_to_plot = data_recon.shape[0]  # num of images
        recon_batch_size = self.batch_size

        labels_predicted = np.zeros(shape=(num_pts_to_plot))
        # labels_predicted_mapped = []
        for b in range(int(np.ceil(num_pts_to_plot * 1.0 / recon_batch_size))):
            if (b + 1) * recon_batch_size > num_pts_to_plot:
                pt_indx = np.arange(b * recon_batch_size, num_pts_to_plot)
            else:
                pt_indx = np.arange(b * recon_batch_size, (b + 1) * recon_batch_size)
            xtrue = data_recon[pt_indx, :]

            zhats_gen, zhats_label = self.sess.run([self.z_infer_gen, self.z_infer_label], feed_dict={self.x: xtrue})

            labels_predicted[pt_indx] = np.argmax(zhats_label, axis=1)
            # print(np.argmax(zhats_label, axis=1))
            # x = np.argmax(zhats_label, axis=1)
            # for value in list(x):
            #     labels_predicted_mapped.append(mode_labels[value])

        # hacer dataframe con columnas y_real, y_pred
        df = pd.DataFrame(columns=['label', 'cluster'])
        df['label'] = label_recon
        df['cluster'] = labels_predicted.astype(np.uint8)
        print(df.head())

        co_mat = pd.crosstab(df.label, df.cluster, normalize='all')
        sn.set(font_scale=1.4)
        sn.heatmap(co_mat)
        plt.savefig('co-ocurrence.png')
        # tengo labels reales y label generadas y el mapeo correspondiente
        # from sklearn.metrics import confusion_matrix
        # y_pred = []
        # for i in range(len(labels_predicted)):
        #     y_pred.append(labels_predicted[i])
        #
        # labels = ['black', 'blue', 'brown', 'green', 'grey', 'orange', 'pink', 'purple', 'red',
        #           'white',
        #           'yellow']
        # cm = confusion_matrix(true_labels_mapped, y_pred)
        # print(cm)
        # import seaborn as sn
        # import pandas as pd
        # df_cm = pd.DataFrame(cm, range(11), range(11))
        # print(df_cm.head())
        # sn.set(font_scale=1.4)  # for label size
        # sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size
        # plt.savefig("cm.png")

    def generate_latent_points(self, latent_dim, n_samples):
        # x_input = np.random.rand(latent_dim * n_samples)
        if self.num_classes in (16, 12, 14):
            batch_size = 1008
        elif self.num_classes == 5:
            batch_size = 1000
        elif self.num_classes in (11, 13):
            batch_size = 1001
        elif self.num_classes == 9:
            batch_size = 1008
        else:
            batch_size = 1000
        ## list of len 1001
        label_index = np.tile(np.arange(self.num_classes), n_samples)

        x_input = self.z_sampler(n_samples, self.z_dim, self.sampler, num_class=self.num_classes,
                                 n_cat=self.n_cat)
        print(f'x_input shape: {x_input.shape}')
        z_input = x_input.reshape(n_samples, latent_dim)
        return z_input

    def plot_generated(self, examples, n):
        print(examples.shape)

        if 'colors' in args.data:
            examples = examples.reshape((examples.shape[0], 32, 32, 3))
            print(examples.shape)
            import cv2
            print(f'CIE-LAB! Bx: {examples.shape[0]}')
            for i in range(examples.shape[0]):
                examples[i] = cv2.cvtColor((examples[i] * 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
                examples[i] = examples[i] / 255
        # plot images
        for i in range(n):
            # define subplot
            plt.subplot(np.ceil(n / 2), np.floor(n / 2), 1 + i)
            # turn off axis
            plt.axis('off')
            # plot raw pixel data
            plt.imshow(examples[i, :, :])
        plt.savefig('interpolation.png')

    # uniform interpolation between two points in latent space
    # def interpolate_points(self, p1, p2, n_steps=10):
    #     # interpolate ratios between the points
    #     ratios = np.linspace(0, 1, num=n_steps)
    #     # linear interpolate vectors
    #     vectors = list()
    #     for ratio in ratios:
    #         v = (1.0 - ratio) * p1 + ratio * p2
    #         vectors.append(v)
    #     return np.asarray(vectors)

    # TODO: interpolar solo la parte one-hot

    def interpolate_points(self, p1, p2, n_steps=10):
        # interpolate ratios between the points
        ratios = np.linspace(0, 1, num=n_steps)
        # linear interpolate vectors
        vectors = list()
        for ratio in ratios:
            v = self.slerp(ratio, p1, p2)
            vectors.append(v)
        return np.asarray(vectors)

    def interpolate_classes(self, c1, c2, n_steps=10):
        points = list()
        for i in range(n_steps + 1):
            mu = (n_steps - i) / n_steps
            p1 = np.hstack((np.zeros(z_dim - self.num_classes), np.eye(self.num_classes)[c1] * mu))
            p2 = np.hstack((np.zeros(z_dim - self.num_classes), np.eye(self.num_classes)[c2] * (1 - mu)))
            p = p1 + p2
            points.append(p)
        return np.asarray(points)

    # spherical linear interpolation (slerp)
    def slerp(self, val, low, high):
        omega = np.arccos(np.clip(np.dot(low / np.linalg.norm(low), high / np.linalg.norm(high)), -1, 1))
        so = np.sin(omega)
        if so == 0:
            # L'Hopital's rule/LERP
            return (1.0 - val) * low + val * high
        return np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high

    def interpolate_latent_space(self):
        # generate points in latent space
        n = 2
        num_points = 10
        # pts = self.generate_latent_points(self.dim_gen + self.num_classes, n)
        # class_1 = np.random.randint(low=0, high=self.num_classes)
        class_1 = 1
        # con el np.eye me quedo la file donde el one-hot se corresponde a la clase
        # pts_1 = np.hstack((0.1 * np.random.randn(1, z_dim - self.num_classes), np.eye(self.num_classes)[class_1]))
        # class_2 = np.random.randint(low=0, high=self.num_classes)
        class_2 = 3
        # pts_2 = np.hstack((0 * np.random.randn(1, z_dim - self.num_classes), np.eye(self.num_classes)[class_2]))
        while class_2 == class_1:
            class_2 = np.random.randint(low=0, high=self.num_classes)

        # label_index = np.random.randint(low=0, high=self.num_classes, size=n)
        # pts = np.hstack((0 * np.random.randn(n, z_dim - self.num_classes), np.eye(self.num_classes)[label_index]))
        # interpolate pairs
        results = None
        interpolated = self.interpolate_classes(class_1, class_2)
        X = self.sess.run(self.x_, feed_dict={self.z: interpolated})
        if results is None:
            results = X
        else:
            results = np.vstack((results, X))
        # for i in range(0, num_points, 2):
        #     # interpolate points in latent space
        #     interpolated = self.interpolate_points(pts[i], pts[i + 1])
        #     X = self.sess.run(self.x_, feed_dict={self.z: interpolated})
        #     if results is None:
        #         results = X
        #     else:
        #         results = np.vstack((results, X))
        # plot the result
        print(f'CLASS 1: {class_1}, CLASS 2: {class_2}')
        self.plot_generated(results, num_points + 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--data', type=str, default='mnist')
    parser.add_argument('--model', type=str, default='clus_wgan')
    parser.add_argument('--sampler', type=str, default='one_hot')
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--dz', type=int, default=30)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--beta_n', type=float, default=10.0)
    parser.add_argument('--beta_c', type=float, default=10.0)
    parser.add_argument('--timestamp', type=str, default='')
    parser.add_argument('--train', type=str, default='False')
    parser.add_argument('--label', type=str, default='False')
    parser.add_argument('--path', type=str, default='')
    parser.add_argument('--modes', type=str, default='False')
    parser.add_argument('--from_checkpoint', type=str, default='False')
    parser.add_argument('--reconstruct', type=str, default='False')
    parser.add_argument('--cm', type=str, default='False')
    parser.add_argument('--interpolate', type=str, default='False')
    parser.add_argument('--ap', type=str, default='False')

    args = parser.parse_args()
    data = importlib.import_module(args.data)
    model = importlib.import_module(args.data + '.' + args.model)

    num_classes = args.K
    dim_gen = args.dz
    n_cat = 1
    batch_size = args.bs
    beta_cycle_gen = args.beta_n
    beta_cycle_label = args.beta_c
    timestamp = args.timestamp

    z_dim = dim_gen + num_classes * n_cat
    d_net = model.Discriminator()
    g_net = model.Generator(z_dim=z_dim)
    enc_net = model.Encoder(z_dim=z_dim, dim_gen=dim_gen)
    # x_sampler
    xs = data.DataSampler()
    # z_sampler
    zs = util.sample_Z

    cl_gan = clusGAN(g_net, d_net, enc_net, xs, zs, args.data, args.model, args.sampler,
                     num_classes, dim_gen, n_cat, batch_size, beta_cycle_gen, beta_cycle_label)
    if args.train == 'True':
        if args.from_checkpoint == 'True':
            cl_gan.load(pre_trained=True)
            timestamp = 'pre-trained'
            cl_gan.train()
        else:
            cl_gan.train()

    else:

        print('Attempting to Restore Model ...')
        if timestamp == '':
            cl_gan.load(pre_trained=True)
            timestamp = 'pre-trained'
        else:
            cl_gan.load(pre_trained=False, timestamp=timestamp)

        if args.label == 'True':
            # print('Labeling query image...')
            # # km, latent = cl_gan.recon_enc(timestamp, val=False)
            # km = None
            # latent = None
            # cl_gan.label_img(args.path, km, latent)
            cl_gan.label_test_images()

        elif args.modes == 'True':
            cl_gan.gen_from_all_modes()
        elif args.reconstruct == 'True':
            bx, bx_labels = xs.test()
            bx = bx[250:252]
            cl_gan.encoder_to_gen(bx)

            bx, bx_labels = xs.test()
            cl_gan.encoder_to_gen(bx)
            if args.data == 'termisk':
                import cv2

                img = cv2.imread('/content/termisk_dataset/test/2/323817_30_0_0.png', cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
                img = np.reshape(img, 28 * 28)
                img = img / 255
                bx = img
                img2 = cv2.imread('/content/termisk_dataset/test/1/51045_270_0_0.png', cv2.IMREAD_GRAYSCALE)
                img2 = cv2.resize(img2, (28, 28), interpolation=cv2.INTER_AREA)
                img2 = np.reshape(img2, 28 * 28)
                img2 = img2 / 255
                bx = np.vstack((bx, img2))
                cl_gan.encoder_to_gen(bx)

        elif args.cm == 'True' and args.data == 'colors_new':
            cl_gan.colors_confusion_matrix()
        elif args.cm == 'True' and args.data == 'mnist':
            cl_gan.mnist_confusion_matrix()
        elif args.cm == 'True':
            cl_gan.co_matrix()
        elif args.ap == 'True':
            mapping = cl_gan.assignment_problem()
            cl_gan.recon_enc(timestamp, val=False, mapping=mapping)
        elif args.interpolate == 'True':
            cl_gan.interpolate_latent_space()
        else:
            cl_gan.recon_enc(timestamp, val=False)
