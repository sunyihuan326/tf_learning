# coding:utf-8 
'''
created on 2019/4/16

@author:sunyihuan
'''
# a = input()
# if a == "":
#     print("Error")
# print(sum(map(int, a.split(" "))))
# from sklearn.cluster import KMeans
import time
# import inspect as ist
# print(ist.getsource(KMeans))


import numpy as np
a = np.random.rand(100000)
b = np.random.rand(100000)
s = time.time()

aa = np.random.rand(2, 3)
print(aa)
print(np.exp(aa))

c = np.dot(a, b)
print("c dot:", c)
e = time.time()
print("dot time: %.2f" % ((e - s) * 1000))

c0 = 0
for i in range(100000):
    c0 += a[i] * b[i]
print("c loop:", c0)

ee = time.time()

print("loop time: {:.2f}".format((ee - s) * 1000))
