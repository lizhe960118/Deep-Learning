import os
# from six.moves import urllib
from urllib.request import urlretrieve
from scipy.io import loadmat
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imresize
# %matplotlib inline


def download_hock(count, block_size, total_size):
    if count % 20 == 0 or count * block_size == total_size:
        percentage = 100.0 * count * block_size / total_size
        barstring = ['=' for _ in range(int(
            percentage / 2.0))] + ['>'] + ['.' for _ in range(50 - int(percentage / 2.0))]
        barstring = '[' + ''.join(barstring) + ']'
        outstring = '%02.02f% % (%02.02f of %02.02f MB) \t\t' + barstring
        print(
            outstring %
            (percentage,
             count *
             block_size /
             1024.0 /
             1024.0,
             total_size /
             1024.0 /
             1024.0),
            end='\r')


path = 'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat'
fname = 'vgg-19.mat'
if not os.path.exists(fname):
    print('Downloading ...')
    file_path, _ = urlretrieve(path, filename=fname, reporthook=download_hock)
    print('DONE.')
if not os.path.exists('content.jpg'):
    urlretrieve('', filename='content.jpg')
    urlretrieve('', filename='style.jpg')

original_layers = loadmat(fname)["layers"][0]
print(original_layers.shape)


def get_layer_name(i):
    return original_layers[i][0][0][0][0]


def get_layer_weights(i):
    return original_layers[i][0][0][2][0][0]


def get_layer_bias(i):
    return original_layers[i][0][0][2][0][1]


def get_layer_params(i):
    return (get_layer_weights(i), get_layer_bias(i))


print(get_layer_name(0))
print(get_layer_weights(0))
print(get_layer_bias(0))
layer_names = [get_layer_name(i) for i in range(len(original_layers))]


def get_layer_by_name(name):
    return layer_names.index(name)


print(','.join(layer_names))
conv_layers = [ln for ln in layer_names if ln.startswith('conv')]
pool_layers = [ln for ln in layer_names if ln.startswith('pool')]
print(conv_layers)

# We are not trying to rebuild the complete model.
#  We will take the convolutional layers that are needed to reason about the image style
# and image content, and we will only work with those layers.


def create_activated_convlayer(prev, i):
    layer_index = get_layer_by_name(conv_layers[i])
    W, b = get_layer_params(layer_index)
    W = tf.constant(W)
    b = tf.constant(np.reshape(b, (b.size)))
    conv = tf.nn.conv2d(prev, filter=W, strides=[1, 2, 2, 1], padding='SAME')
    return tf.nn.relu(conv)


def create_pool_layer(prev):
    return tf.nn.avg_pool(
        prev, ksize=[
            1, 2, 2, 1], strides=[
            1, 2, 2, 1], padding='SAME')


def get_next_convlayer(i, prev):
    next_i = i + 1
    return (next_i, create_activated_convlayer(prev, i))


def get_next_convlayer_name(i):
    return conv_layers[i]


def get_last_convlayer_name(i):
    return conv_layers[i - 1]


tf.reset_default_graph()

model = {}

content = plt.imread('content.jpg')
style = plt.imread('style.jpg')

scale_down = 1

height = content.shape[0] // scale_down
width = content.shape[1] // scale_down
# height = 300
# width = 450

index = 0

model['in'] = tf.Variable(np.zeros((1, height, width, 3)), dtype=tf.float32)
index, model[get_next_convlayer_name(
    index - 1)] = get_next_convlayer(index, model['in'])
index, model[get_next_convlayer_name(
    index - 1)] = get_next_convlayer(index, model[get_last_convlayer_name(index)])
model['avgpool1'] = create_pool_layer(model[get_last_convlayer_name(index)])
index, model[get_next_convlayer_name(
    index - 1)] = get_next_convlayer(index, model['avgpool1'])
index, model[get_next_convlayer_name(
    index - 1)] = get_next_convlayer(index, model[get_last_convlayer_name(index)])
model['avgpool2'] = create_pool_layer(model[get_last_convlayer_name(index)])
index, model[get_next_convlayer_name(
    index - 1)] = get_next_convlayer(index, model['avgpool2'])
for i in range(3):
    index, model[get_next_convlayer_name(
        index - 1)] = get_next_convlayer(index, model[get_last_convlayer_name(index)])
model['avgpool3'] = create_pool_layer(model[get_last_convlayer_name(index)])
index, model[get_next_convlayer_name(
    index - 1)] = get_next_convlayer(index, model['avgpool3'])
for i in range(3):
    index, model[get_next_convlayer_name(
        index - 1)] = get_next_convlayer(index, model[get_last_convlayer_name(index)])
model['avgpool4'] = create_pool_layer(model[get_last_convlayer_name(index)])
index, model[get_next_convlayer_name(
    index - 1)] = get_next_convlayer(index, model['avgpool4'])
for i in range(3):
    index, model[get_next_convlayer_name(
        index - 1)] = get_next_convlayer(index, model[get_last_convlayer_name(index)])
model['avgpool5'] = create_pool_layer(model[get_last_convlayer_name(index)])

print(model)

means = np.reshape([116.779, 123.68, 103.939], (1, 1, 3))

# print(content.shape)
# print(means.shape)
# print(content)
# print(means)

def preprocess_img(img_in):
    img = img_in.astype("float32")
    img = imresize(img, [height, width])
    img = img - means
    img = img[np.newaxis]
    return img


def unprocess_img(img_in):
    img = img_in
    img = img[0]
    img = img + means
    img = np.clip(img, 0, 255).astype('uint8')
    print(img)
    return img


processed_content = preprocess_img(content)
processed_style = preprocess_img(style)
unprocessed_content = unprocess_img(processed_content)
unprocessed_style = unprocess_img(processed_style)

# plt.figure(figsize=(10, 10))
# plt.axis('off')
# plt.imshow(unprocessed_content.astype('uint8'))
# plt.show()
#
# plt.figure(figsize=(10, 10))
# plt.axis('off')
# plt.imshow(unprocessed_style.astype('uint8'))
# plt.show()

content_layer = 'conv4_2'
style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
style_weights = [.8, .8, .8, .8, .8]

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
content_features = sess.run(
    model[content_layer], feed_dict={
        model['in']: processed_content})
style_features = sess.run([model[sl] for sl in style_layers], feed_dict={
                          model['in']: processed_style})

# define loss function


def content_loss(p):
    size = np.prod(content_features.shape[1:])
    return (1 / (2.0 * size)) * \
        tf.reduce_sum(tf.pow((p - content_features), 2))


def gram_matrix(features, n, m):
    features_t = tf.reshape(features, (m, n))
    return tf.matmul(tf.transpose(features_t), features_t)


def style_loss(a, x):
    n = a.shape[3]
    m = a.shape[1] * a.shape[2]
    a_matrix = gram_matrix(a, n, m)
    g_matrix = gram_matrix(x, n, m)
    return (1 / (4 * n ** 2 * m ** 2)) * \
        tf.reduce_sum(tf.pow((g_matrix - a_matrix), 2))


def var_loss(x):
    h, w = x.get_shape().as_list()[1], x.get_shape().as_list()[2]
    dx = tf.square(x[:, :h - 1, :w - 1, :] - x[:, :h - 1, 1:, :])
    dy = tf.square(x[:, :h - 1, :w - 1, :] - x[:, 1:, :w - 1, :])
    return tf.reduce_sum(tf.pow((dx + dy), 1.25))


e = [style_loss(sf, model[ln]) for sf, ln in zip(style_features, style_layers)]
styleloss = sum([style_weights[l] * e[l] for l in range(len(style_layers))])
contentloss = content_loss(model[content_layer])
varloss = var_loss(model['in'])

alpha = 1
beta = 500
gamma = 0.1

total_loss = alpha * contentloss + beta * styleloss + gamma * varloss

noise_ratio = 0.7
content_ratio = 1. - noise_ratio
noise = np.random.uniform(-15, 15, processed_content.shape)
input_image = (processed_content * content_ratio) + noise_ratio * noise

unp = unprocess_img(input_image)
plt.imshow(unp.astype('uint8'))
plt.show()

optimizer = tf.train.AdamOptimizer(1).minimize((total_loss))
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
# writer = tf.summary.FileWriter("./logs",sess.graph)
sess.run(model['in'].assign(input_image))
pass

# train
for i in range(1000):
    # if i % 1000 == 0:
    #
    #     m_in = sess.run(model['in'])
    #     plt.imshow(unprocess_img(m_in).astype('uint8'))
    #     plt.show()
    _, ls = sess.run([optimizer, total_loss])

    if i % 50 == 0:
        print(i, ls)
        # writer.add_summary(ls,i) #将日志数据写入文件
# writer.close()

m_in = sess.run(model['in'])
plt.imshow(unprocess_img(m_in).astype('uint8'))
plt.savefig("transfered.jpg")
plt.show()