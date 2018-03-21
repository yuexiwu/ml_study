from logistic_regression.core import *

data = pd.read_csv('ex2data1.txt', names=['exam1', 'exam2', 'admitted'])
data.head()  # 看前五行
data.describe()
sns.set(context="notebook", style="darkgrid", palette=sns.color_palette("RdBu", 2))

sns.lmplot('exam1', 'exam2', hue='admitted', data=data,
           size=6,
           fit_reg=False,
           scatter_kws={"s": 50}
           )
plt.show()
# 看下数据的样子

X = get_X(data)
print(X.shape)

y = get_y(data)
print(y.shape)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(np.arange(-10, 10, step=0.01),
        sigmoid(np.arange(-10, 10, step=0.01)))
ax.set_ylim((-0.1, 1.1))
ax.set_xlabel('z', fontsize=18)
ax.set_ylabel('g(z)', fontsize=18)
ax.set_title('sigmoid function', fontsize=18)
# plt.show()

theta = np.zeros(3)  # X(m*n) so theta is n*1
print(gradient(theta, X, y))

res = opt.minimize(fun=cost, x0=theta, args=(X, y), method='Newton-CG', jac=gradient)
print(res)

final_theta = res.x
y_pred = predict(X, final_theta)

print(classification_report(y, y_pred))
print(res.x)

coef = -(res.x / res.x[2])
print(coef)
x = np.arange(130, step=0.1)
y = coef[0] + coef[1] * x

print(data.describe())

sns.set(context="notebook", style="ticks", font_scale=1.5)

sns.lmplot('exam1', 'exam2', hue='admitted', data=data,
           size=6,
           fit_reg=False,
           scatter_kws={"s": 25}
           )

plt.plot(x, y, 'grey')
plt.xlim(0, 130)
plt.ylim(0, 130)
plt.title('Decision Boundary')
plt.show()
