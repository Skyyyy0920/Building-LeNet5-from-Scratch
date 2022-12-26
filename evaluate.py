# coding=utf-8
import numpy as np


def predict(test_images, theta):
    """
    to predict the class in test dataset
    :param test_images: [10000 * 784]
    :param theta: [10 * 784]
    :return: predict result [10000 * 10]
    """
    scores = np.dot(test_images, theta.T)  # [10000, 10]
    predict = np.argmax(scores, axis=1)  # [10000, 1]
    return predict


def cal_accuracy(y_pred, y):
    # TODO: Compute the accuracy among the test set and store it in acc
    acc = 0
    for i in range(y.shape[0]):
        if y[i] != y_pred[i]:
            acc += 1
    return 1 - acc / y.shape[0]
