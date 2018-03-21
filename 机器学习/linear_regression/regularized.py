from linear_regression.core import *
import pandas as pd
import seaborn as sns

sns.set(context="notebook", style="whitegrid", palette="dark")
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# 正则化
raw_data = pd.read_csv('ex1data2.txt', names=['square', 'bedrooms', 'price'])
raw_data.head()
data = normalize_feature(raw_data)
print(data.head())

X = get_x(data)
print(X.shape, type(X))

y = get_y(data)
print(y.shape, type(y))  # 看下数据的维度和类型

alpha = 0.01  # 学习率
theta = np.zeros(X.shape[1])  # X.shape[1]：特征数n
epoch = 500  # 轮数
final_theta, cost_data = batch_gradient_decent(theta, X, y, epoch, alpha=alpha)
sns.tsplot(time=np.arange(len(cost_data)), data=cost_data)
plt.xlabel('epoch', fontsize=18)
plt.ylabel('cost', fontsize=18)
plt.show()
print(final_theta)

# 学习率

base = np.logspace(-1, -5, num=4)
candidate = np.sort(np.concatenate((base, base * 3)))
print('学习率')
print(candidate)
epoch = 50

fig, ax = plt.subplots(figsize=(16, 9))

for alpha in candidate:
    _, cost_data = batch_gradient_decent(theta, X, y, epoch, alpha=alpha)
    ax.plot(np.arange(epoch + 1), cost_data, label=alpha)

ax.set_xlabel('epoch', fontsize=18)
ax.set_ylabel('cost', fontsize=18)
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax.set_title('learning rate', fontsize=18)


# plt.show()

#

# 正规方程
def normalEqn(X, y):
    theta = np.linalg.inv(X.T @ X) @ X.T @ y  # X.T@X等价于X.T.dot(X)
    return theta


final_theta2 = normalEqn(X, y)  # 感觉和批量梯度下降的theta的值有点差距
print(final_theta2)

