# coding=utf-8
import numpy as np

from data_process import data_convert
from softmax_regression import softmax_regression


def train(train_images, train_labels, k, iters=5, alpha=0.5):
    m, n = train_images.shape  # 60000, 784
    # data processing
    x, y = data_convert(train_images, train_labels, m, k)  # x:[m, n] -> [60000, 784], y:[m, k] -> [60000, 10]

    # Initialize theta.  Use a matrix where each column corresponds to a class,
    # and each row is a classifier coefficient for that class.
    theta = np.random.rand(k, n)  # [k, n] -> [10, 784]
    # do the softmax regression
    theta = softmax_regression(theta, x, y, iters, alpha)
    return theta
