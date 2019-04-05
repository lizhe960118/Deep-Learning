# pool_height, pool_width, pool_stride =  pool_param['pool_height'],pool_param['pool_width'],pool_param['pool_stride']
# H_out = int((H - pool_height)/ pool_stride + 1) 
# W_out = int( (W - pool_width) / pool_stride + 1)  
# out = np.zeros(N, C, H_out, W_out)
#  for i in range(H_out):
#      for j in range(W_out):
#      	x_padded_mask = x[:, :, i * pool_stride: i * pool_stride + pool_height, j * pool_stride:j * pool_stride + pool_width]
#      	out[:,:,i,j] = np.max(x_padded_mask, axis=(2, 3))

# max_mask = np.max(x_padded_mask, axis=(2, 3))
# temp_binary_mask = (x_padded_mask == (max_mask[:,:, None, None]))

# dx[:, :, i * pool_stride: i * pool_stride + pool_height, j * pool_stride:j * pool_stride + pool_width] = (dout[:, :,i, j])[:,:, None, None] * temp_binary_mask

# import numpy as np

# a = np.array([[0,1,0,0],
	# [1,0,0,0],[0,0,0,1],[0,0,1,0]])

# b = (a == 1)
# print(b, type(b))

# c = a[b]
# print(c, type(c))
# print(a[0])

"""
a = np.random.randn(5, 3)
print(a)
b = a[:5, :]
print(b)
c = b[2]
print(c)
print(c.shape)
d = np.random.randn(5, 3)
print(np.dot(d, c))
"""

# li = [3, 2, 5, 7, 8, 1, 5]

# # li[-1], li[li.index(min(li))] = li[li.index(min(li))], li[-1]
# print(li)
# print(max(li))
# print(li.index(max(li)))
# a = li[li.index(max(li))]
# print(a)
# # li[0], li[4] = li[4], li[0]
# # li = [3, 2, 5, 7, 8, 1, 5]
# li[0], li[li.index(max(li))] = li[li.index(max(li))], li[0]
# print(li)

# from collections import Counter
# tasks = ["A","A","A","B","B"]
# tasks_dict = Counter(tasks)
# print(tasks_dict)
# tasks_dict = sorted(tasks_dict.items(), key = lambda e:e[1], reverse=True)
# print(tasks_dict)

# tasks = [1,2,3,5,4]
# tasks.sort()
# print(tasks)

labels = [[y for i in range(10)] for y in range(16)]
print(labels)
import numpy as np
# y = np.append(labels)
y = np.array(labels)
print(y)
y = y.flatten()
print(y)
print(y.shape[0])
y = y.flatten()
print(y)

y = [np.random.randint(0, 9) for i in range(1000)]
print(y)

import random
y = random.sample(range(16000), 20)
print(y)

y = "generate_{}.t7".format(100)
print(y)