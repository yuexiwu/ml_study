import pandas as pd
import seaborn as sns

sns.set(context="notebook", style="whitegrid", palette="dark")
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


def get_x(df):
    ones = pd.DataFrame({'ones': np.ones(len(df))})  # ones是m行1列的dataframe
    data = pd.concat([ones, df], axis=1)  # 合并数据，根据列合并
    return data.iloc[:, :-1].as_matrix()  # 这个操作返回 ndarray,不是矩阵


def get_y(df):  # 读取标签
    #     '''assume the last column is the target'''
    return np.array(df.iloc[:, -1])  # df.iloc[:, -1]是指df的最后一列


# 计算代价函数
def lr_cost(theta, X, y):
    #     """
    #     X: R(m*n), m 样本数, n 特征数
    #     y: R(m)
    #     theta : R(n), 线性回归的参数
    #     """
    m = X.shape[0]  # m为样本数

    inner = X @ theta - y  # R(m*1)，X @ theta等价于X.dot(theta)

    # 1*m @ m*1 = 1*1 in matrix multiplication
    # but you know numpy didn't do transpose in 1d array, so here is just a
    # vector inner product to itselves
    square_sum = inner.T @ inner
    cost = square_sum / (2 * m)

    return cost


# 批量梯度下降
def gradient(theta, X, y):
    m = X.shape[0]

    inner = X.T @ (X @ theta - y)  # (m,n).T @ (m, 1) -> (n, 1)，X @ theta等价于X.dot(theta)

    return inner / m


def gradient(theta, X, y):
    m = X.shape[0]

    inner = X.T @ (X @ theta - y)  # (m,n).T @ (m, 1) -> (n, 1)，X @ theta等价于X.dot(theta)

    return inner / m


def batch_gradient_decent(theta, X, y, epoch, alpha=0.01):
    #   拟合线性回归，返回参数和代价
    #     epoch: 批处理的轮数
    #     """
    cost_data = [lr_cost(theta, X, y)]
    _theta = theta.copy()  # 拷贝一份，不和原来的theta混淆

    for _ in range(epoch):
        _theta = _theta - alpha * gradient(_theta, X, y)
        cost_data.append(lr_cost(_theta, X, y))

    return _theta, cost_data


# 批量梯度下降函数
# 特征缩放
def normalize_feature(df):
    #     """Applies function along input axis(default 0) of DataFrame."""
    return df.apply(lambda column: (column - column.mean()) / column.std())


