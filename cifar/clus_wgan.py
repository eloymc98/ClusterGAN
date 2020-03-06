import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.contrib.layers as tcl


def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)


class Discriminator(object):
    def __init__(self, x_dim=3072):
        self.x_dim = x_dim
        self.name = 'cifar/clus_wgan/d_net'

    # Every time the discriminator is called, we obtain a new output
    def __call__(self, x, reuse=True):
        # Variable scope allows you to create new variables and to share already created ones
        # while providing checks to not create or share by accident
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            bs = tf.shape(x)[0]
            # Reshape to 3d
            print(f'Discriminator x before: {x.shape}')

            x = tf.reshape(x, [bs, 32, 32, 3])
            print(f'Discriminator x after: {x.shape}')

            # output = conv2d(input, num_outputs, kernel_size, stride)
            # tf.indetity -> Return a tensor with the same shape and contents as input
            conv1 = tc.layers.convolution2d(
                x, 64, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )
            conv1 = leaky_relu(conv1)
            print(f'Discriminator conv1: {conv1.shape}')

            conv2 = tc.layers.convolution2d(
                conv1, 128, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )
            conv2 = tc.layers.batch_norm(conv2)
            conv2 = leaky_relu(conv2)
            print(f'Discriminator conv2: {conv2.shape}')

            conv3 = tc.layers.convolution2d(
                conv2, 256, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )
            conv3 = tc.layers.batch_norm(conv3)
            conv3 = leaky_relu(conv3)
            print(f'Discriminator conv3: {conv3.shape}')

            conv4 = tc.layers.convolution2d(
                conv3, 512, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )
            conv4 = tc.layers.batch_norm(conv4)
            conv4 = leaky_relu(conv4)
            print(f'Discriminator conv4 before: {conv4.shape}')

            conv4 = tcl.flatten(conv4)
            print(f'Discriminator conv4 after: {conv4.shape}')

            # Flatten conv2 output to use it as input for fc.

            # fc = fully_connected(input, num_outputs)
            fc1 = tc.layers.fully_connected(conv4, 1, activation_fn=tf.identity)
            print(f'Discriminator fc1: {fc1.shape}')

            return fc1

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class Generator(object):
    def __init__(self, z_dim=50, x_dim=3072):
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.name = 'cifar/clus_wgan/g_net'

    def __call__(self, z):
        with tf.variable_scope(self.name) as vs:
            bs = tf.shape(z)[0]
            print(f'Generator bs : {bs}')
            print(f'Generator z: {z.shape}')

            fc1 = tc.layers.fully_connected(
                z, 2 * 2 * 448,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )

            # out: 1792
            print(f'Generator fc1 before: {fc1.shape}')

            fc1 = tf.reshape(fc1, tf.stack([bs, 2, 2, 448]))
            fc1 = tc.layers.batch_norm(fc1)
            fc1 = tf.nn.relu(fc1)
            print(f'Generator fc1 after: {fc1.shape}')

            conv1 = tc.layers.convolution2d_transpose(
                fc1, 256, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )
            conv1 = tc.layers.batch_norm(conv1)
            conv1 = tf.nn.relu(conv1)
            # out: 16 * 16 * 64
            print(f'Generator conv1: {conv1.shape}')

            conv2 = tc.layers.convolution2d_transpose(
                conv1, 128, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )
            conv2 = tc.layers.batch_norm(conv2)
            conv2 = tf.nn.relu(conv2)
            print(f'Generator conv2: {conv2.shape}')

            conv3 = tc.layers.convolution2d_transpose(
                conv2, 64, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )
            conv3 = tc.layers.batch_norm(conv3)
            conv3 = tf.nn.relu(conv3)
            print(f'Generator conv3: {conv3.shape}')

            conv4 = tc.layers.convolution2d_transpose(
                conv3, 3, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.sigmoid
            )
            print(f'Generator conv4 before: {conv4.shape}')

            conv4 = tf.reshape(conv4, tf.stack([bs, self.x_dim]))
            print(f'Generator conv4 after: {conv4.shape}')

            return conv4

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class Encoder(object):
    def __init__(self, z_dim=50, dim_gen=10, x_dim=3072):
        self.z_dim = z_dim
        self.dim_gen = dim_gen
        self.x_dim = x_dim
        self.name = 'cifar/clus_wgan/enc_net'

    def __call__(self, x, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            bs = tf.shape(x)[0]
            print(f'Encoder x before: {x.shape}')

            x = tf.reshape(x, [bs, 32, 32, 3])
            print(f'Encoder x after: {x.shape}')

            conv1 = tc.layers.convolution2d(
                x, 64, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )
            conv1 = leaky_relu(conv1)
            print(f'Encoder conv1: {conv1.shape}')

            # out: 16 * 16 * 64
            conv2 = tc.layers.convolution2d(
                conv1, 128, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )
            conv2 = tc.layers.batch_norm(conv2)
            conv2 = leaky_relu(conv2)
            # out: 8 * 8 * 128
            print(f'Encoder conv2: {conv2.shape}')


            conv3 = tc.layers.convolution2d(
                conv2, 256, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )
            conv3 = tc.layers.batch_norm(conv3)
            conv3 = leaky_relu(conv3)
            # out: 4 * 4 * 256
            print(f'Encoder conv3: {conv3.shape}')

            conv4 = tc.layers.convolution2d(
                conv3, 512, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )
            # out: 2 * 2 * 512

            conv4 = tc.layers.batch_norm(conv4)
            conv4 = leaky_relu(conv4)
            print(f'Encoder conv4 before: {conv4.shape}')

            conv4 = tcl.flatten(conv4)
            print(f'Encoder conv4 after: {conv4.shape}')

            # out = 2048

            fc1 = tc.layers.fully_connected(conv4, self.z_dim, activation_fn=tf.identity)
            print(f'Encoder fc1: {fc1.shape}')

            # out = n_classes + z_n

            logits = fc1[:, self.dim_gen:]
            y = tf.nn.softmax(logits)
            return fc1[:, 0:self.dim_gen], y, logits

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]
