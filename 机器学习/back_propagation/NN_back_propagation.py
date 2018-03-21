from back_propagation.core import *

X, _ = load_data('ex4data1.mat')
plot_100_image(X)
plt.show()

X_raw, y_raw = load_data('ex4data1.mat', transpose=False)
X = np.insert(X_raw, 0, np.ones(X_raw.shape[0]), axis=1)  # 增加全部为1的一列
print(X.shape)
print(y_raw)

y = expand_y(y_raw)
t1, t2 = load_weight('ex4weights.mat')
print(t1.shape, t2.shape)

theta = serialize(t1, t2)  # 扁平化参数，25*401+10*26=10285
print(theta.shape)

_, _, _, _, h = feed_forward(theta, X)
print(h)
# 5000*10
print(cost(theta, X, y))

print(regularized_cost(theta, X, y))

print(X.shape, y.shape)
print(t1.shape, t2.shape)
print(theta.shape)

print(sigmoid_gradient(0))

d1, d2 = deserialize(gradient(theta, X, y))
print(d1.shape, d2.shape)

# gradient_checking(theta, X, y, epsilon=0.0001)
# 这个运行很慢，谨慎运行
gradient_checking(theta, X, y, epsilon=0.0001, regularized=True)
# 这个运行很慢，谨慎运行

res = nn_training(X, y)  # 慢
print(res)

_, y_answer = load_data('ex4data1.mat')
print(y_answer[:20])
final_theta = res.x

plot_hidden_layer(final_theta)
plt.show()