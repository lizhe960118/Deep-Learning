# 使用tensorflow学习简单计算图的表示
import tensorflow as tf 
tf.reset_default_graph()

a = tf.Variable(1, name = 'a', trainable = True)
b = tf.constant(2, name = 'b')

c = a + b
assign = tf.assign(a, 5)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	_, out_c = sess.run([assign, c])
	print(_, out_c)

# tf.Print 
two_node = tf.constant(2)
three_node = tf.constant(3)
sum_node = two_node + three_node
# this new copy of two_node is not on the computation path, so nothing prints!
print_two_node = tf.Print(two_node, [two_node, three_node, sum_node])
sess = tf.Session()
print(sess.run(sum_node)) # 只会得到5，tf.Print不在计算路径上
# print(sess.run(print_two_node)) # [2][3][5]