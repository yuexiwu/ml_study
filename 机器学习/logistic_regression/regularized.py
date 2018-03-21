from logistic_regression.core import *

df = pd.read_csv('ex2data2.txt', names=['test1', 'test2', 'accepted'])
print(df.head())
sns.set(context="notebook", style="ticks", font_scale=1.5)

sns.lmplot('test1', 'test2', hue='accepted', data=df,
           size=6,
           fit_reg=False,
           scatter_kws={"s": 50}
           )

plt.title('Regularized Logistic Regression')
plt.show()

# 特征映射

x1 = np.array(df.test1)
x2 = np.array(df.test2)

data = feature_mapping(x1, x2, power=6)
print(data.shape)
data.head()

print(data.describe())

# 正则化代价函数
theta = np.zeros(data.shape[1])
X = feature_mapping(x1, x2, power=6, as_ndarray=True)
print(X.shape)

y = get_y(df)
print(y.shape)

print(regularized_cost(theta, X, y, l=1))
print(regularized_gradient(theta, X, y))

res = opt.minimize(fun=regularized_cost, x0=theta, args=(X, y), method='Newton-CG', jac=regularized_gradient)
print(res)

final_theta = res.x
y_pred = predict(X, final_theta)

print(classification_report(y, y_pred))

# 使用不同的λ

print(draw_boundary(power=6, l=1))

# 过拟合
draw_boundary(power=6, l=0)  # no regularization, over fitting，#lambda=0,没有正则化，过拟合了

# 欠拟合

draw_boundary(power=6, l=100)  # underfitting，#lambda=100,欠拟合
