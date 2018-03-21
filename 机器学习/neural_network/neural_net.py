from neural_network.core import *

theta1, theta2 = load_weight('ex3weights.mat')

print(theta1.shape, theta2.shape)
X, y = load_data('ex3data1.mat', transpose=False)

X = np.insert(X, 0, values=np.ones(X.shape[0]), axis=1)  # intercept

print(X.shape, y.shape)

a1 = X
z2 = a1 @ theta1.T  # (5000, 401) @ (25,401).T = (5000, 25)
print(z2.shape)

z2 = np.insert(z2, 0, values=np.ones(z2.shape[0]), axis=1)
a2 = sigmoid(z2)
print(z2.shape)

z3 = a2 @ theta2.T
print(z3.shape)
a3 = sigmoid(z3)
print(a3)

y_pred = np.argmax(a3, axis=1) + 1  # numpy is 0 base index, +1 for matlab convention，返回沿轴axis最大值的索引，axis=1代表行
print(y_pred.shape)
print(classification_report(y, y_pred))
