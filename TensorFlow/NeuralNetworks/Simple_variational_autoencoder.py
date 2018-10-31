# Simple_VAE.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data')

tf.reset_default_graph()

batch_size = 64
X_in = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28], name="X")
Y = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28], name='Y')
Y_flat = tf.reshape(Y, shape=[-1, 28 * 28])
keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')

dec_in_channels = 1
n_latent = 8
reshaped_dim = [-1, 7, 7, dec_in_channels]

inputs_decoder = 49 * dec_in_channels // 2


def lrelu(x, alpha=0.3):
    return tf.maximum(x, tf.multiply(x, alpha))


def encoder(X_in, keep_prob):
    activation = lrelu
    with tf.variable_scope('encoder', reuse=None):
        X = tf.reshape(X_in, shape=[-1, 28, 28, 1])
        conv1 = tf.layers.conv2d(
            X,
            filters=64,
            kernel_size=4,
            strides=2,
            padding="same",
            activation=activation)
        conv1_drop = tf.nn.dropout(conv1, keep_prob=keep_prob)
        conv2 = tf.layers.conv2d(
            conv1_drop,
            filters=64,
            kernel_size=4,
            strides=2,
            padding="same",
            activation=activation)
        conv2_drop = tf.nn.dropout(conv2, keep_prob=keep_prob)
        conv3 = tf.layers.conv2d(
            conv2_drop,
            filters=64,
            kernel_size=4,
            strides=1,
            padding="same",
            activation=activation)
        conv3_drop = tf.nn.dropout(conv3, keep_prob=keep_prob)
        x_out = tf.contrib.layers.flatten(conv3_drop)
        mn = tf.layers.dense(x_out, units=n_latent)
        sd = 0.5 * tf.layers.dense(x_out, units=n_latent)
        epsilon = tf.random_normal(tf.stack([tf.shape(x_out)[0], n_latent]))
        z = mn + tf.multiply(epsilon, tf.exp(sd))
        return z, mn, sd

# decode尝试重建输入图像，用到转置卷积


def decoder(sampled_z, keep_prob):
    with tf.variable_scope("decoder", reuse=None):
        dense1 = tf.layers.dense(
            sampled_z,
            units=inputs_decoder,
            activation=lrelu)
        dense2 = tf.layers.dense(
            dense1,
            units=inputs_decoder * 2 + 1,
            activation=lrelu)
        reshaped_x = tf.reshape(dense2, reshaped_dim)
        conv_tran1 = tf.layers.conv2d_transpose(
            reshaped_x,
            filters=64,
            kernel_size=4,
            strides=2,
            padding="same",
            activation=tf.nn.relu)
        conv_tran1_drop = tf.nn.dropout(conv_tran1, keep_prob=keep_prob)
        conv_tran2 = tf.layers.conv2d_transpose(
            conv_tran1_drop,
            filters=64,
            kernel_size=4,
            strides=1,
            padding="same",
            activation=tf.nn.relu)
        conv_tran2_drop = tf.nn.dropout(conv_tran2, keep_prob=keep_prob)
        conv_tran3 = tf.layers.conv2d_transpose(
            conv_tran2_drop,
            filters=64,
            kernel_size=4,
            strides=1,
            padding="same",
            activation=tf.nn.relu)
        x_flat = tf.contrib.layers.flatten(conv_tran3)
        x_dec = tf.layers.dense(
            x_flat,
            units=28 * 28,
            activation=tf.nn.sigmoid)
        img = tf.reshape(x_dec, shape=[-1, 28, 28])
        return img


# 将encoder和decoder联系在一起
sampled_z, mn, sd = encoder(X_in, keep_prob)
dec = decoder(sampled_z, keep_prob)

# 计算损失函数，并将隐藏值从高斯分布中采样
unreshaped = tf.reshape(dec, [-1, 28 * 28])
img_loss = tf.reduce_sum(tf.squared_difference(unreshaped, Y_flat), 1)
latent_loss = -0.5 * \
    tf.reduce_sum(1.0 + 2.0 * sd - tf.square(mn) - tf.exp(2.0 * sd), 1)
loss = tf.reduce_mean(img_loss + latent_loss)
optimizer = tf.train.AdamOptimizer(0.005).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 训练网络
for i in range(30000):
    batch = [np.reshape(b, [28, 28])
             for b in mnist.train.next_batch(batch_size=batch_size)[0]]
    sess.run(optimizer, feed_dict={X_in: batch, Y: batch, keep_prob: 0.8})
    if not i % 200:
        ls, d, i_ls, d_ls, mu, sigm = sess.run([loss, dec, img_loss, latent_loss, mn, sd], feed_dict={
                                               X_in: batch, Y: batch, keep_prob: 1.0})
        plt.imshow(np.reshape(batch[0], [28, 28]), cmap="gray")
        plt.show()
        plt.imshow(d[0], cmap='gray')
        plt.show()
        print(i, ls, np.mean(i_ls), np.mean(d_ls))

# 生成新数据：只需要从单位正态分布里面采集一个值，输入到解码器。
randoms = [np.random.normal(0, 1, n_latent) for _ in range(10)]
imgs = sess.run(dec, feed_dict={sampled_z: randoms, keep_prob: 1.0})
imgs = [np.reshape(imgs[i], [28, 28]) for i in range(len(imgs))]

for img in imgs:
    plt.figure(figsize=(1.1))
    plt.axis("off")
    plt.imshow(img, cmap='gray')
    plt.show()
