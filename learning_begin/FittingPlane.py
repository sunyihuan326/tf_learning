# coding:utf-8 
'''
fitting a plane
the first course in W3School for tensorflow

created on 2019/4/8

@author:sunyihuan
'''
import tensorflow as tf
import numpy as np

x_data = np.float32(np.random.rand(2, 100))
y_data = np.dot([0.1, 0.2], x_data) + 0.3

w = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))

y = tf.matmul(w, x_data) + b

loss = tf.reduce_mean(tf.square(y - y_data))
optimze = tf.train.GradientDescentOptimizer(0.5)
train = optimze.minimize(loss)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for step in range(1, 401):
        sess.run(train)
        if step % 20 == 0:
            print(step, sess.run(w), sess.run(b))
