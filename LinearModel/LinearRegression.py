import numpy as np 
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt 

#NumPy用numpy.random.Generator随机数

#创建随机数生成实例
rng = np.random.default_rng( seed = 0 )
'''
rng1 = np.random.default_rng( seed = 0 )                       #默认PCG64作为位生成器
rng2 = np.random.default_rng( np.random.MT19937 (seed = 0) )   #指定MT9937作为位生成器
'''

#生成随机数
x = 2 * rng.random((100,1))
y = 4 + 3 * x + rng.normal( loc = 0 , scale = 1 , size = (100,1))

plt.scatter(x,y)
plt.xlabel('x')
plt.ylabel('y')
plt.title("Random Data For Linear Regression")
plt.savefig("/home/victorwu/MachineLearning/output/ScatterPlotLR.png")
plt.close()

#加载线性回归
model = LinearRegression()

# 拟合模型
model.fit( x , y )

# 输出模型的参数
print(f"斜率 (w): {model.coef_[0][0]}")
print(f"截距 (b): {model.intercept_[0]}")

y_pred = model.predict(x)

# 作出拟合直线
plt.scatter(x, y)
plt.plot(x, y_pred, color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression Fit')
plt.savefig("/home/victorwu/MachineLearning/output/NumpyLinearRegression.png")