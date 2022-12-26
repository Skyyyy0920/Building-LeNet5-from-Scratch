# coding=utf-8
import numpy as np


def softmax_regression(theta, x, y, iters, alpha):
    # TODO: Do the softmax regression by computing the gradient and the objective function value of every iteration
    #  and update the theta
    """
    softmax regression
    :param theta: Parameter matrix [10 * 784]
    :param x: train images [60000 * 784]
    :param y: train labels [60000 * 10]
    :param iters: iterations
    :param alpha: learning rate
    :return: matrix of theta, loss record
    """
    loss = []
    iterations = []
    record = range(0, iters, 25)
    for i in range(iters):
        pred = np.dot(theta, x.T)  # [10 * 60000]
        pred_exp = np.exp(pred)
        pred = pred_exp / np.sum(pred_exp, axis=0)  # [10 * 60000] = [10 * 60000] / [1 * 60000]
        pred = pred.T  # [60000 * 10]
        los = -np.multiply(np.log(pred), y).sum() / x.shape[0]  # [60000 * 10]
        gradient = np.dot(x.T, (pred-y))  # [784 * 10] = [784 * 60000] * [60000 * 10]
        print(np.max(gradient), np.min(gradient))
        theta = theta - alpha * gradient.T  # update parameter matrix
        print("{}th epoch:".format(i), los)
        if i in record:  # 每25轮输出一次loss
            loss.append(los)
            iterations.append(i)

    return theta, iterations, loss
    
