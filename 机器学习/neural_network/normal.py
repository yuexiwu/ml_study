from neural_network.core import *

X, y = load_data('ex3data1.mat')

print(X.shape)
print(y.shape)
pick_one = np.random.randint(0, 5000)
plot_an_image(X[pick_one, :])
plt.show()
print('this should be {}'.format(y[pick_one]))

plot_100_image(X)
plt.show()

raw_X, raw_y = load_data('ex3data1.mat')
print(raw_X.shape)
print(raw_y.shape)

X = np.insert(raw_X, 0, values=np.ones(raw_X.shape[0]), axis=1)  # 插入了第一列（全部为1）
print(X.shape)

# y have 10 categories here. 1..10, they represent digit 0 as category 10 because matlab index start at 1
# I'll ditit 0, index 0 again
y_matrix = []

for k in range(1, 11):
    y_matrix.append((raw_y == k).astype(int))

# last one is k==10, it's digit 0, bring it to the first position，最后一列k=10，都是0，把最后一列放到第一列
y_matrix = [y_matrix[-1]] + y_matrix[:-1]
y = np.array(y_matrix)

print(y.shape)

# 扩展 5000*1 到 5000*10
#     比如 y=10 -> [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]: ndarray
#
print(y)
t0 = logistic_regression(X, y[0])
print(t0.shape)
y_pred = predict(X, t0)
print('Accuracy={}'.format(np.mean(y[0] == y_pred)))

# 可怕 0.9974的成功率

k_theta = np.array([logistic_regression(X, y[k]) for k in range(10)])
print(k_theta.shape)

prob_matrix = sigmoid(X @ k_theta.T)
np.set_printoptions(suppress=True)
print(prob_matrix)
y_pred = np.argmax(prob_matrix, axis=1)  # 返回沿轴axis最大值的索引，axis=1代表行
print(y_pred)
y_answer = raw_y.copy()
y_answer[y_answer == 10] = 0
print(classification_report(y_answer, y_pred))
