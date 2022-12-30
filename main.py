# coding=utf-8
import numpy as np
import struct
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from data_processing import *
from train import *
from evaluate import *
from utils_function import *
from LeNet5 import *

if __name__ == '__main__':
    # 显示所有列
    pd.set_option('display.max_columns', None)
    # 显示所有行
    pd.set_option('display.max_rows', None)
    # 设置value的显示长度
    pd.set_option('max_colwidth', 100)
    # 设置1000列时才换行
    pd.set_option('display.width', 1000)

    # get the data
    train_images, train_labels, test_images, test_labels = load_data()
    print("Got data...\n")

    k = 10
    iters = 50
    alpha = 0.000018

    # data processing, normalization&zero-padding
    train_images = normalize(zero_pad(train_images[:, :, :, np.newaxis], 2), 'LeNet5')
    test_images = normalize(zero_pad(test_images[:, :, :, np.newaxis], 2), 'LeNet5')
    print("The shape of training image with padding:", train_images.shape)
    print("The shape of testing image with padding: ", test_images.shape)

    # train LeNet5
    start_time = time.time()
    LeNet5 = LeNet5()
    train(LeNet5, train_images, train_labels)
    print("Finished training...\n")

    end_time = time.time()
    print("The total training time is {}s\n".format(end_time - start_time))

    # evaluate on the test dataset
    y_predict = predict(test_images, theta)
    accuracy = cal_accuracy(y_predict, test_labels)
    print("The accuracy is: ", accuracy)
    print("Finished test...")

    # plt.plot(iterations, loss)
    # plt.grid(True, linestyle='--', alpha=0.5)  # 添加网格信息，默认是True，风格设置为虚线，alpha为透明度
    # plt.title('loss-iteration')
    # plt.show()
