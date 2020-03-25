import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.contrib.layers as tcl


def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)


class Discriminator(object):
    # x_dim = 28*28
    def __init__(self, x_dim=4096):
        self.x_dim = x_dim
        self.name = 'termisk/clus_wgan/d_net'

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

            x = tf.reshape(x, [bs, 64, 64, 1])
            print(f'Discriminator x after: {x.shape}')
            conv1 = tc.layers.convolution2d(
                x, 32, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
            )
            conv1 = leaky_relu(conv1)
            # out = (bs,32,32,32)

            # output = conv2d(input, num_outputs, kernel_size, stride)
            # tf.indetity -> Return a tensor with the same shape and contents as input
            conv2 = tc.layers.convolution2d(
                conv1, 64, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
            )
            conv2 = leaky_relu(conv2)
            # out = (bs,16,16,64)
            print(f'Discriminator conv2: {conv2.shape}')

            conv3 = tc.layers.convolution2d(
                conv2, 128, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
            )
            conv3 = leaky_relu(conv3)
            # out = (bs,8,8,128)
            print(f'Discriminator conv3 before: {conv3.shape}')
            conv3 = tcl.flatten(conv3)
            # out = (bs,8192)
            print(f'Discriminator conv3 after: {conv3.shape}')
            # Flatten conv2 output to use it as input for fc.

            # fc = fully_connected(input, num_outputs)
            fc1 = tc.layers.fully_connected(
                conv3, 2048,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
            )
            fc1 = leaky_relu(fc1)
            print(f'Discriminator fc1: {fc1.shape}')
            fc2 = tc.layers.fully_connected(fc1, 1, activation_fn=tf.identity)
            print(f'Discriminator fc2: {fc2.shape}')
            return fc2

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class Generator(object):
    def __init__(self, z_dim=10, x_dim=4096):
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.name = 'termisk/clus_wgan/g_net'

    def __call__(self, z):
        with tf.variable_scope(self.name) as vs:
            bs = tf.shape(z)[0]
            print(f'Generator bs : {bs}')
            print(f'Generator z: {z.shape}')

            fc1 = tc.layers.fully_connected(
                z, 2048,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )
            fc1 = tc.layers.batch_norm(fc1)
            fc1 = tf.nn.relu(fc1)
            print(f'Generator fc1: {fc1.shape}')
            fc2 = tc.layers.fully_connected(
                fc1, 8 * 8 * 128,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )
            print(f'Generator fc2 before: {fc2.shape}')
            fc2 = tf.reshape(fc2, tf.stack([bs, 8, 8, 128]))
            fc2 = tc.layers.batch_norm(fc2)
            fc2 = tf.nn.relu(fc2)
            print(f'Generator fc2 after: {fc2.shape}')
            # out = (bs,8,8,128)
            conv1 = tc.layers.convolution2d_transpose(
                fc2, 64, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )
            conv1 = tc.layers.batch_norm(conv1)
            conv1 = tf.nn.relu(conv1)
            print(f'Generator conv1: {conv1.shape}')
            # out = (bs,16,16,64)
            conv2 = tc.layers.convolution2d_transpose(
                conv1, 32, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.sigmoid
            )
            conv2 = tc.layers.batch_norm(conv2)
            conv2 = tf.nn.relu(conv2)
            print(f'Generator conv2: {conv2.shape}')
            # out = (bs,32,32,32)
            conv3 = tc.layers.convolution2d_transpose(
                conv2, 1, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.sigmoid
            )
            conv3 = tc.layers.batch_norm(conv3)
            conv3 = tf.nn.relu(conv3)
            # out = (bs,64,64,1)
            conv3 = tf.reshape(conv3, tf.stack([bs, self.x_dim]))
            print(f'Generator conv2 after: {conv2.shape}')
            return conv3

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class Encoder(object):
    def __init__(self, z_dim=10, dim_gen=10, x_dim=4096):
        self.z_dim = z_dim
        self.dim_gen = dim_gen
        self.x_dim = x_dim
        self.name = 'termisk/clus_wgan/enc_net'

    def __call__(self, x, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            bs = tf.shape(x)[0]
            print(f'Encoder x before: {x.shape}')
            x = tf.reshape(x, [bs, 64, 64, 1])
            print(f'Encoder x after: {x.shape}')
            conv1 = tc.layers.convolution2d(
                x, 32, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
            )
            conv1 = leaky_relu(conv1)
            # out = (bs,32,32,32)
            conv2 = tc.layers.convolution2d(
                conv1, 64, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )
            conv2 = leaky_relu(conv2)
            print(f'Encoder conv2: {conv2.shape}')
            # out = (bs,16,16,64)
            conv3 = tc.layers.convolution2d(
                conv2, 128, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )
            conv3 = leaky_relu(conv3)
            # out = (bs,8,8,128)
            print(f'Encoder conv3 before: {conv3.shape}')
            conv3 = tcl.flatten(conv3)
            print(f'Encoder conv3 after: {conv3.shape}')
            # out = (bs,8192)
            fc1 = tc.layers.fully_connected(
                conv3, 2048,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )
            fc1 = leaky_relu(fc1)
            print(f'Encoder fc1: {fc1.shape}')
            fc2 = tc.layers.fully_connected(fc1, self.z_dim, activation_fn=tf.identity)
            print(f'Encoder fc2: {fc2.shape}')
            logits = fc2[:, self.dim_gen:]
            y = tf.nn.softmax(logits)
            return fc2[:, 0:self.dim_gen], y, logits

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


# class Discriminator(object):
#     # x_dim = 28*28
#     def __init__(self, x_dim=784):
#         self.x_dim = x_dim
#         self.name = 'termisk/clus_wgan/d_net'
#
#     # Every time the discriminator is called, we obtain a new output
#     def __call__(self, x, reuse=True):
#         # Variable scope allows you to create new variables and to share already created ones
#         # while providing checks to not create or share by accident
#         with tf.variable_scope(self.name) as vs:
#             if reuse:
#                 vs.reuse_variables()
#             bs = tf.shape(x)[0]
#             # Reshape to 3d
#             x = tf.reshape(x, [bs, 28, 28, 1])
#             # output = conv2d(input, num_outputs, kernel_size, stride)
#             # tf.indetity -> Return a tensor with the same shape and contents as input
#             conv1 = tc.layers.convolution2d(
#                 x, 64, [4, 4], [2, 2],
#                 weights_initializer=tf.random_normal_initializer(stddev=0.02),
#                 weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
#                 activation_fn=tf.identity
#             )
#             conv1 = leaky_relu(conv1)
#
#             conv2 = tc.layers.convolution2d(
#                 conv1, 128, [4, 4], [2, 2],
#                 weights_initializer=tf.random_normal_initializer(stddev=0.02),
#                 weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
#                 activation_fn=tf.identity
#             )
#             conv2 = tc.layers.batch_norm(conv2)
#             conv2 = leaky_relu(conv2)
#
#             conv3 = tc.layers.convolution2d(
#                 conv2, 256, [4, 4], [2, 2],
#                 weights_initializer=tf.random_normal_initializer(stddev=0.02),
#                 weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
#                 activation_fn=tf.identity
#             )
#             conv3 = tc.layers.batch_norm(conv3)
#             conv3 = leaky_relu(conv3)
#
#             conv4 = tc.layers.convolution2d(
#                 conv3, 512, [4, 4], [2, 2],
#                 weights_initializer=tf.random_normal_initializer(stddev=0.02),
#                 weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
#                 activation_fn=tf.identity
#             )
#             conv4 = tc.layers.batch_norm(conv4)
#             conv4 = leaky_relu(conv4)
#             conv4 = tcl.flatten(conv4)
#             # Flatten conv2 output to use it as input for fc.
#
#             # fc = fully_connected(input, num_outputs)
#             fc1 = tc.layers.fully_connected(conv4, 1, activation_fn=tf.identity)
#             return fc1
#
#     @property
#     def vars(self):
#         return [var for var in tf.global_variables() if self.name in var.name]
#
#
# class Generator(object):
#     def __init__(self, z_dim=50, x_dim=784):
#         self.z_dim = z_dim
#         self.x_dim = x_dim
#         self.name = 'termisk/clus_wgan/g_net'
#
#     def __call__(self, z):
#         with tf.variable_scope(self.name) as vs:
#             bs = tf.shape(z)[0]
#             print(f'Generator bs : {bs}')
#             fc1 = tc.layers.fully_connected(
#                 z, 2 * 2 * 448,
#                 weights_initializer=tf.random_normal_initializer(stddev=0.02),
#                 weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
#                 activation_fn=tf.identity
#             )
#             fc1 = tf.reshape(fc1, tf.stack([bs, 2, 2, 448]))
#             fc1 = tc.layers.batch_norm(fc1)
#             fc1 = tf.nn.relu(fc1)
#
#             conv1 = tc.layers.convolution2d_transpose(
#                 fc1, 256, [4, 4], [2, 2],
#                 weights_initializer=tf.random_normal_initializer(stddev=0.02),
#                 weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
#                 activation_fn=tf.identity
#             )
#             conv1 = tc.layers.batch_norm(conv1)
#             conv1 = tf.nn.relu(conv1)
#             conv2 = tc.layers.convolution2d_transpose(
#                 conv1, 128, [4, 4], [2, 2],
#                 weights_initializer=tf.random_normal_initializer(stddev=0.02),
#                 weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
#                 activation_fn=tf.identity
#             )
#             conv2 = tc.layers.batch_norm(conv2)
#             conv2 = tf.nn.relu(conv2)
#
#             conv3 = tc.layers.convolution2d_transpose(
#                 conv2, 64, [4, 4], [2, 2],
#                 weights_initializer=tf.random_normal_initializer(stddev=0.02),
#                 weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
#                 activation_fn=tf.identity
#             )
#             conv3 = tc.layers.batch_norm(conv3)
#             conv3 = tf.nn.relu(conv3)
#
#             conv4 = tc.layers.convolution2d_transpose(
#                 conv3, 3, [4, 4], [2, 2],
#                 weights_initializer=tf.random_normal_initializer(stddev=0.02),
#                 weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
#                 activation_fn=tf.sigmoid
#             )
#
#             conv4 = tf.reshape(conv4, tf.stack([bs, self.x_dim]))
#             return conv4
#
#     @property
#     def vars(self):
#         return [var for var in tf.global_variables() if self.name in var.name]
#
#
# class Encoder(object):
#     def __init__(self, z_dim=50, dim_gen=10, x_dim=784):
#         self.z_dim = z_dim
#         self.dim_gen = dim_gen
#         self.x_dim = x_dim
#         self.name = 'termisk/clus_wgan/enc_net'
#
#     def __call__(self, x, reuse=True):
#         with tf.variable_scope(self.name) as vs:
#             if reuse:
#                 vs.reuse_variables()
#             bs = tf.shape(x)[0]
#             x = tf.reshape(x, [bs, 28, 28, 1])
#             conv1 = tc.layers.convolution2d(
#                 x, 64, [4, 4], [2, 2],
#                 weights_initializer=tf.random_normal_initializer(stddev=0.02),
#                 weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
#                 activation_fn=tf.identity
#             )
#             conv1 = leaky_relu(conv1)
#
#             conv2 = tc.layers.convolution2d(
#                 conv1, 128, [4, 4], [2, 2],
#                 weights_initializer=tf.random_normal_initializer(stddev=0.02),
#                 weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
#                 activation_fn=tf.identity
#             )
#             conv2 = tc.layers.batch_norm(conv2)
#             conv2 = leaky_relu(conv2)
#
#             conv3 = tc.layers.convolution2d(
#                 conv2, 256, [4, 4], [2, 2],
#                 weights_initializer=tf.random_normal_initializer(stddev=0.02),
#                 weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
#                 activation_fn=tf.identity
#             )
#             conv3 = tc.layers.batch_norm(conv3)
#             conv3 = leaky_relu(conv3)
#
#             conv4 = tc.layers.convolution2d(
#                 conv3, 512, [4, 4], [2, 2],
#                 weights_initializer=tf.random_normal_initializer(stddev=0.02),
#                 weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
#                 activation_fn=tf.identity
#             )
#             conv4 = tc.layers.batch_norm(conv4)
#             conv4 = leaky_relu(conv4)
#             conv4 = tcl.flatten(conv4)
#
#             fc1 = tc.layers.fully_connected(conv4, self.z_dim, activation_fn=tf.identity)
#             logits = fc1[:, self.dim_gen:]
#             y = tf.nn.softmax(logits)
#             return fc1[:, 0:self.dim_gen], y, logits
#
#     @property
#     def vars(self):
#         return [var for var in tf.global_variables() if self.name in var.name]
