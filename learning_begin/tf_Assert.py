# coding:utf-8 
'''
TensorFlow 中与 Assert 相关的函数进行具体的举例说明,断言给定条件的真假与条件中持有的元素

created on 2019/4/8

@author:sunyihuan
'''

import tensorflow as tf

X = tf.constant([2., 3])

assert_op = tf.Assert(tf.less_equal(tf.reduce_max(X), 1.), [X])
with tf.control_dependencies([assert_op]):
    with tf.Session() as sess:
        print(sess.run(X))


with tf.control_dependencies([tf.assert_positive(X)]):
    output = tf.reduce_sum(X)
    with tf.Session() as sess1:
        print(sess1.run(output))
