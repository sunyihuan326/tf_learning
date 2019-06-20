# coding:utf-8 
'''
created on 2019/4/18

@author:sunyihuan
'''
import numpy as np

A = np.array([[1, 2, 3, 4], [3, 4, 5, 6], [7, 0, 3, 5]])
cal = A.sum(axis=0)
print(cal)

p = 100 * A / cal.reshape(1, 4)
print(p)
p = 100 * A / cal[:-1].reshape(3, 1)
print(p)

B = np.random.rand(5)

