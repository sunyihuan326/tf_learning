# coding:utf-8 
'''
created on 2019/4/20

@author:sunyihuan
'''

import numpy as np
import math


class conv2d(object):
    '''
    numpy实现卷积
    '''

    def __init__(self, features, kernels, channels, stride, padding):
        self.output = self.conv(features, kernels, channels, stride, padding=padding)

    def conv(self, features, kernels, channels, stride, padding="same"):
        '''
        卷积forward
        :param features: 输入特征
        :param kernels: 卷积核大小
        :param channels: filter通道数
        :param stride: 移动步长
        :param padding: 是否padding
        :return:
        '''

        (w, h, c) = np.array(features).shape

        if padding == "same":
            p = math.ceil(((w - 1) * stride - w + kernels) / 2)

            input = np.pad(features, ((p, p), (p, p), (0, 0)), "constant")
        else:
            input = features
        (w, h, c) = np.array(input).shape

        output_w = int(math.floor((w - kernels) / stride + 1))
        output_h = int(math.floor((h - kernels) / stride + 1))
        output_c = channels
        output = np.random.rand(output_w, output_h, output_c)
        for cc in range(channels):
            weigth, b = self.weigths(input, kernels)
            for ww in range(output_w):
                for hh in range(output_h):
                    end = np.sum(
                        input[ww * stride:ww * stride + kernels, hh * stride:hh * stride + kernels, :] * weigth) + b

                    output[ww][hh][cc] = int(end)
        return output

    def weigths(self, featu, kernels):
        '''
        生成参数w
        :param featu:input尺寸
        :param kernels:
        :return:
        '''
        assert len(np.array(featu).shape) == 3

        (w, h, c) = np.array(featu).shape
        weigth = np.random.rand(kernels, kernels, c)
        b = np.zeros(1)

        return weigth, b


if __name__ == "__main__":
    inputs_features = np.random.rand(5, 5, 3)
    conv2 = conv2d(inputs_features, 3, 4, 2, "1").output
    print(conv2.shape)
