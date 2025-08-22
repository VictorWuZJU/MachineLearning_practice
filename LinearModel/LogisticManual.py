import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#加载数据集
iris = load_iris()
X = iris.data
y = iris.target

#添加偏置列，给最后一列后加上偏置
X = np.column_stack([X, np.ones((X.shape[0], 1))])

y_onehot = np.eye(3)[y] #one-hot编码

X_tr, X_te, y_tr, y_te, y_tr_oh, y_te_oh = train_test_split(X, y, y_onehot, test_size=0.2, random_state=42, stratify=y)

d = X_tr.shape[1]   # 特征维度（含偏置列）
C = len(np.unique(y))   # 类别数

#参数初始化
np.random.seed(42)  #设置随机种子
W = np.random.randn(d, C) * 0.01   # (d, C)型状的随机正态分布N(0,1e-4)

#softmax函数
def softmax(Z):
    exp = np.exp(Z - np.max(Z, axis=1, keepdims=True))  # 数值稳定:softmax的平移不变性
    return exp / np.sum(exp, axis=1, keepdims=True)

#交叉熵损失函数
def cross_entropy(Y_hat, Y):
    m = Y.shape[0]
    return -np.sum(Y * np.log(Y_hat + 1e-15)) / m

#计算梯度
def gradient(X, Y, Y_hat):
    return X.T @ (Y_hat - Y) / X.shape[0] 

#梯度下降
lr = 0.2  #学习率
epochs = 5000
for epoch in range(epochs):
    Z = X_tr @ W              # (n, C)
    Y_hat = softmax(Z)
    loss = cross_entropy(Y_hat, y_tr_oh)
    grad = gradient(X_tr, y_tr_oh, Y_hat)
    W -= lr * grad

#输出评估数据
    if (epoch+1) % 500 == 0:
        preds = np.argmax(Y_hat, axis=1)
        acc_tr = np.mean(preds == y_tr)
        print(f"epoch {epoch+1:4d}  loss={loss:.4f}  acc={acc_tr:.4f}")
Z_te = X_te @ W
y_pred = np.argmax(softmax(Z_te), axis=1)
acc_te = np.mean(y_pred == y_te)
print("Test accuracy:", acc_te)