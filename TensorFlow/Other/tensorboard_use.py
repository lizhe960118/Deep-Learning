import tensorflow as tf
import numpy as np


# activation_function=None线性函数
def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
    layer_name = "layer%s" % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal(
                [in_size, out_size]))  # Weight中都是随机变量
            tf.summary.histogram(layer_name + "/weights", Weights)  # 可视化观看变量
        with tf.name_scope('biases'):
            biases = tf.Variable(
                tf.zeros([1, out_size]) + 0.1)  # biases推荐初始值不为0
            tf.summary.histogram(layer_name + "/biases", biases)  # 可视化观看变量
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + \
                biases  # inputs*Weight+biases
            tf.summary.histogram(
                layer_name + "/Wx_plus_b",
                Wx_plus_b)  # 可视化观看变量
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        tf.summary.histogram(layer_name + "/outputs", outputs)  # 可视化观看变量
        return outputs


# 创建数据x_data，y_data
# [-1,1]区间，300个单位，np.newaxis增加维度
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)  # 噪点
y_data = np.square(x_data) - 0.5 + noise
with tf.name_scope('inputs'):  # 结构化
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

# 三层神经，输入层（1个神经元），隐藏层（10神经元），输出层（1个神经元）
l1 = add_layer(xs, 1, 10, n_layer=1, activation_function=tf.nn.relu)  # 隐藏层
prediction = add_layer(l1, 10, 1, n_layer=2, activation_function=None)  # 输出层

# prediction值与y_data差别
with tf.name_scope('loss'):
    loss = tf.reduce_mean(
        tf.reduce_sum(
            tf.square(
                ys -
                prediction),
            reduction_indices=[1]))  # square()平方,sum()求和,mean()平均值
    tf.summary.scalar('loss', loss)  # 可视化观看常量

with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(
        0.1).minimize(loss)  # 0.1学习效率,minimize(loss)减小loss误差

# init = tf.initialize_all_variables()
init = tf.global_variables_initializer()
sess = tf.Session()
# 合并到Summary中
merged = tf.summary.merge_all()
# 选定可视化存储目录
writer = tf.summary.FileWriter("logs", sess.graph)

sess.run(init)  # 先执行init
# 训练1k次
for i in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        result = sess.run(
            merged,
            feed_dict={
                xs: x_data,
                ys: y_data})  # merged也是需要run的
        print(i, loss)
writer.add_summary(result, i)  # result是summary类型的，需要放入writer中，i步数（x轴）

# import tensorflow as tf
# import numpy as np
#
# # 输入数据
# x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
# noise = np.random.normal(0, 0.05, x_data.shape)
# y_data = np.square(x_data) - 0.5 + noise
#
# # 输入层
# with tf.name_scope('input_layer'):  # 输入层。将这两个变量放到input_layer作用域下，tensorboard会把他们放在一个图形里面
#     xs = tf.placeholder(
#         tf.float32, [
#             None, 1], name='x_input')  # xs起名x_input，会在图形上显示
#     ys = tf.placeholder(
#         tf.float32, [
#             None, 1], name='y_input')  # ys起名y_input，会在图形上显示
#
# # 隐层
# with tf.name_scope('hidden_layer'):  # 隐层。将隐层权重、偏置、净输入放在一起
#     with tf.name_scope('weight'):  # 权重
#         W1 = tf.Variable(tf.random_normal([1, 10]))
#         tf.summary.histogram('hidden_layer/weight', W1)
#     with tf.name_scope('bias'):  # 偏置
#         b1 = tf.Variable(tf.zeros([1, 10]) + 0.1)
#         tf.summary.histogram('hidden_layer/bias', b1)
#     with tf.name_scope('Wx_plus_b'):  # 净输入
#         Wx_plus_b1 = tf.matmul(xs, W1) + b1
#         tf.summary.histogram('hidden_layer/Wx_plus_b', Wx_plus_b1)
# output1 = tf.nn.relu(Wx_plus_b1)
#
# # 输出层
# with tf.name_scope('output_layer'):  # 输出层。将输出层权重、偏置、净输入放在一起
#     with tf.name_scope('weight'):  # 权重
#         W2 = tf.Variable(tf.random_normal([10, 1]))
#         tf.summary.histogram('output_layer/weight', W2)
#     with tf.name_scope('bias'):  # 偏置
#         b2 = tf.Variable(tf.zeros([1, 1]) + 0.1)
#         tf.summary.histogram('output_layer/bias', b2)
#     with tf.name_scope('Wx_plus_b'):  # 净输入
#         Wx_plus_b2 = tf.matmul(output1, W2) + b2
#         tf.summary.histogram('output_layer/Wx_plus_b', Wx_plus_b2)
# output2 = Wx_plus_b2
#
# # 损失
# with tf.name_scope('loss'):  # 损失
#     loss = tf.reduce_mean(
#         tf.reduce_sum(
#             tf.square(
#                 ys - output2),
#             reduction_indices=[1]))
#     tf.summary.scalar('loss', loss)
# with tf.name_scope('train'):  # 训练过程
#     train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#
# # 初始化
# init = tf.global_variables_initializer()
# sess = tf.Session()
# sess.run(init)
# merged = tf.summary.merge_all()  # 将图形、训练过程等数据合并在一起
# writer = tf.summary.FileWriter('logs', sess.graph)  # 将训练日志写入到logs文件夹下
#
# # 训练
# for i in range(1000):
#     sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
#     if(i % 50 == 0):  # 每50次写一次日志
#         result = sess.run(
#             merged,
#             feed_dict={
#                 xs: x_data,
#                 ys: y_data})  # 计算需要写入的日志数据
#         print(i, loss)
#         writer.add_summary(result, i)  # 将日志数据写入文件
#
# writer.close()