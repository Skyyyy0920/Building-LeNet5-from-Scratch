# coding=utf-8
import numpy as np
import struct
import os
import time
from data_process import load_mnist, load_data
from train import train
from evaluate import predict, cal_accuracy
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 显示所有列
    pd.set_option('display.max_columns', None)
    # 显示所有行
    pd.set_option('display.max_rows', None)
    # 设置value的显示长度
    pd.set_option('max_colwidth', 100)
    # 设置1000列时才换行
    pd.set_option('display.width', 1000)

    # initialize the parameters needed
    data_dir = "dataset"
    train_data_dir = "train-images.idx3-ubyte"
    train_label_dir = "train-labels.idx1-ubyte"
    test_data_dir = "t10k-images.idx3-ubyte"
    test_label_dir = "t10k-labels.idx1-ubyte"
    k = 10
    iters = 50
    alpha = 0.000018

    start_time = time.time()
    # get the data
    train_images, train_labels, test_images, test_labels = load_data(data_dir, train_data_dir, train_label_dir,
                                                                     test_data_dir, test_label_dir)
    print("Got data. ")

    # train the classifier
    theta, iterations, loss = train(train_images, train_labels, k, iters, alpha)
    print("Finished training. ")

    end_time = time.time()
    print("The total training time is {}s".format(end_time-start_time))

    # evaluate on the test dataset
    y_predict = predict(test_images, theta)
    accuracy = cal_accuracy(y_predict, test_labels)
    print("The accuracy is: ", accuracy)
    print("Finished test. ")

    plt.plot(iterations, loss)
    plt.grid(True, linestyle='--', alpha=0.5)  # 添加网格信息，默认是True，风格设置为虚线，alpha为透明度
    plt.title('loss-iteration')
    plt.show()
