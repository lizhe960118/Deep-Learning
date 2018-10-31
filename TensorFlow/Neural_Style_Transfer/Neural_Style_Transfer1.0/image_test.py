import  matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imresize,imsave

content = plt.imread('content.jpg')
print(content.shape)

plt.axis('off')     #不显示坐标尺寸
plt.imshow(content)
plt.show()

scale_down = 1

height = content.shape[0] // scale_down
width = content.shape[1] // scale_down
# height = 300
# width = 450
print(height, width)
img_in = content
img = imresize(img_in, [height, width])
img = np.reshape(img, (1, height, width, 3))
means = [116.779, 123.68, 103.93]
img = img - means
img = img + means
img = img[0]
img = np.clip(img, 0, 255).astype(np.uint8)
unprocessed_content = img

scale = 5
figure = plt.figure(figsize=(width * scale / height, scale))
plt.axis('off')     #不显示坐标尺寸
plt.imshow(unprocessed_content)
           # .astype('uint8'))
plt.show()
imsave('test.jpg', unprocessed_content)

# means = np.reshape([116.779, 123.68, 103.939], (1, 1, 3))
# means_2 = np.reshape([116.779, 123.68, 103.939], (1, 1, 1, 3))
# means = [116.779, 123.68, 103.93]
# means = [128.0, 128.0, 128.0]
# print("原图像尺寸为:" + str(content.shape))
# print("减去的平均值为：" + str(means.shape))
# print("原图数据为:" + str(content))

# def preprocess_img(img_in):
#     # img = img_in.astype("float32")
#     img = imresize(img_in, [height, width])
#     img = np.reshape(img, (1, height, width, 3))
#     img = img - means
#     print(img)
#     # img = img[np.newaxis]
#     return img
#
#
# def unprocess_img(img_in):
#     img = img_in
#     img = img + means
#     print(img)
#     img = img[0]
#     print(img)
#     img = np.clip(img, 0, 255).astype(np.uint8)
#     print(img)
#     return img


# processed_content = preprocess_img(content)
# unprocessed_content = unprocess_img(processed_content)

# plt.figure(figsize=(10, 10))
# plt.axis('off')
# plt.imshow(content)
# plt.show()
