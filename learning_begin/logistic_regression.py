# coding:utf-8 
'''
tf 实现logistic回归

created on 2019/6/3

@author:sunyihuan
'''
import tensorflow as tf
import numpy as np
from matplotlib import pyplot

# 数据准备
x_data = np.linspace(-0.5, 0.5, 500)
x_data = x_data.reshape([x_data.shape[0], 1])
noise = np.random.normal(0, 0.01, x_data.shape)
y_data = np.square(x_data) + noise

# 模型搭建--输入、输出
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

# 中间层
Weights_1 = tf.Variable(tf.random_normal([1, 20]))
biases_1 = tf.Variable(tf.zeros([1, 20]))
layer_1 = tf.matmul(x, Weights_1) + biases_1
a_1 = tf.nn.tanh(layer_1)

# 输出层
Weights_2 = tf.Variable(tf.random_normal([20, 1]))
biases_2 = tf.Variable(tf.zeros([1, 1]))
layer_2 = tf.matmul(a_1, Weights_2) + biases_2
predictions = tf.nn.tanh(layer_2)

loss = tf.reduce_mean(tf.square(predictions - y))

train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(200):
        _, prediction = sess.run([train_op, predictions], feed_dict={x: x_data, y: y_data})

pyplot.figure()
pyplot.scatter(x_data, y_data)  # 散点是真实值
pyplot.plot(x_data, prediction, 'r-', lw=5)  # 曲线是预测值
pyplot.show()
