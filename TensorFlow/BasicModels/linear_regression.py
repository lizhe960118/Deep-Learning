import tensorflow as tf 
"""
two_node = tf.constant(2)
three_node = tf.constant(3)
sum_node = two_node + three_node
sess = tf.Session()
print(sess.run(sum_node))
# 尽可能少调用sess.run()，在一个sess.run()的调用中返回多个项目
"""
"""
# 占位符，接受外部输入的节点
# 使用feed_dict提供一个值
input_place_holder = tf.placeholder(tf.int32)
sess = tf.Session()
print(sess.run(input_place_holder, feed_dict={input_place_holder:2}))
"""

"""
# 计算路径
input_place_holder = tf.placeholder(tf.int32)
three_node = tf.constant(3)
sum_node = input_place_holder + three_node
sess = tf.Session()
print(sess.run(input_place_holder, feed_dict = {input_place_holder:2}))
print(sess.run(three_node))
print(sess.run(sum_node, feed_dict = {input_place_holder:2}))
"""

"""
# 变量:在运行时保持不变的节点更新值
count_variable = tf.get_variable('count', [])
# tf.get_variable(name, shape) name 是唯一标志这个变量对象的字符串
# shape 是指变量的维数
zero_node = tf.constant(0.)
assign_node = tf.assign(count_variable, zero_node)
# 将值放入变量 
# tf.assign(target, value) 使用value替换target中的值
sess = tf.Session()
sess.run(assign_node)
print(sess.run(count_variable))
"""
"""
# 使用初始化器更新变量
const_init_node = tf.constant_initializer(0.)
count_variable = tf.get_variable("count", [], initializer = const_init_node)
init = tf.global_variables_initializer()
# 这是一个带有副作用的节点，
# 查看全局图并自动将依赖关系添加到图中的每个 tf.initializer
sess = tf.Session()
sess.run(init)
# 通过会话使用const_init_node去更新变量
print(sess.run([count_variable]))
"""
"""
典型的深度学习训练的步骤:
1.获取 输入x 和 真实的输出(true_output)y
2.根据 输入x 预测输出 y_(predict_output)
3.计算真实与预测的差别 losss
4.根据损失的梯度更新 参数(variables)

# 设置参数 w,b
w = tf.get_variable('w', [], initializer = tf.constant_initializer(0.))
b = tf.get_variable('b', [], initializer = tf.constant_initializer(0.))
init = global_variables_initializer()

# 接受输入x和输出y
input_place_holder = tf.placeholder(tf.float32)
output_place_holder = tf.placeholder(tf.float32)
x = input_place_holder()
y = ouput_place_holder()
y_predict = w * x + b

# 计算损失
loss = tf.square(y - y_predict)

# 调用优化器优化loss
optimizer = tf.train.GradientDescentOptimizer(1e-3)
train_op = optimizer.minimize(loss)
 # 将一个节点添加到图中，并将一个指针存储在变量 train_op 中

# 开始会话
sess = tf.Session()
sess.run(init)

import random
true_w = random.random()
true_b = random.random()

# 开始训练
for updata_i in range(1000000):
	input_data = random.random()
	output_data = true_w * input_data + true_b

	_loss, _ = sess.run([loss, train_op], feed_dict = {input_place_holder: input_data, outpout_place_holder: output_datat})
	print(updata_i, _loss)
"""
