from linear_regression.core import *
import pandas as pd
import seaborn as sns

sns.set(context="notebook", style="whitegrid", palette="dark")
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

df = pd.read_csv('ex1data1.txt', names=['population', 'profit'])  # 读取数据并赋予列名
# print(df)
# print(df.head())
# print(df.info())
sns.lmplot('population', 'profit', df, size=6, fit_reg=True)

# plt.show()


X = get_x(df)
print(X.shape, type(X))
Y = get_y(df)
print(Y.shape, type(Y))
theta = np.zeros(X.shape[1])
# 特征数

print(theta)
print(lr_cost(theta, X, Y))

epoch = 500
# 梯度下降
final_theta, cost_data = batch_gradient_decent(theta, X, Y, epoch)
print(final_theta)
print(cost_data)
print(lr_cost(final_theta, X, Y))

# 数据可视化

ax = sns.tsplot(cost_data, time=np.arange(epoch + 1))
ax.set_xlabel('epoch')
ax.set_ylabel('cost')

plt.show()
# 可以看到从第二轮代价数据变换很大，接下来平稳了
