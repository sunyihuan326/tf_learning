# coding:utf-8 
'''

Basic operation for tf

created on 2019/4/8

@author:sunyihuan
'''
import tensorflow as tf

matrix1 = tf.constant([[1.0, 2.0]])
matrix2 = tf.constant([[2.0], [3.0]])

p = tf.matmul(matrix1, matrix2)

sess = tf.Session()

res = sess.run(p)
print(res)

sess.close()  # 关闭会话，释放资源

# with 代码块自动关闭///////////
# 整个模块与
# sess = tf.Session()
# res = sess.run(p)
# print(res)
# sess.close()
# 等价
with tf.Session() as sess:
    result = sess.run([p])
    print(result)

# 使用tf.InteractiveSession()来构建会话的时候，
# 我们可以先构建一个session然后再定义操作（operation），
# 如果我们使用tf.Session()来构建会话我们需要在会话构建之前定义好全部的操作（operation）然后再构建会话
sess = tf.InteractiveSession()
x = tf.Variable([1.0, 2.0])
a = tf.constant([3.0, 3.0])
x.initializer.run()
sub = tf.subtract(x, a)
print(sub.eval())

with tf.Session() as sess0:
    with tf.device("/cpu:0"):  # 指定op的gpu，也可以指定cpu，如："/cpu:0"
        matrix1 = tf.constant([[3., 3.]])
        matrix2 = tf.constant([[2.], [2.]])
        product = tf.matmul(matrix1, matrix2)

state = tf.Variable(0, name="counter")
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)  # 把state的值变为new_value

# 初始化所有变量
init_op = tf.global_variables_initializer()

with tf.Session() as sess1:
    sess1.run(init_op)
    print(sess1.run(state))
    for _ in range(3):
        sess1.run(update)
        print(sess1.run(state))

# feed数据，
input1 = tf.placeholder(tf.float32, [1, 2])  # 如果没有正确提供 feed, placeholder() 操作将会产生错误
input2 = tf.placeholder(tf.float32, [2, 1])
o = tf.matmul(input1, input2)
with tf.Session() as sess2:
    print(sess2.run([o], feed_dict={input1: [[7., 8.]], input2: [[2.], [3]]}))
