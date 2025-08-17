import numpy as np 
from matplotlib import pyplot as plt 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns 
from sklearn.metrics import confusion_matrix

iris = load_iris()
X = iris.data 
y = iris.target

#用train_test_split方法分割数据集形成训练集和测试集
X_train , X_test , y_train , y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#加载逻辑回归模型
model = LogisticRegression ()

#用训练集训练模型
model.fit( X_train , y_train )

#在测试集上测试模型
y_pred = model.predict( X_test )

# 绘制散点图
plt.figure(figsize=(10, 6))
for i, target_name in enumerate(iris.target_names):
    # 绘制实际类别,叉号代表实际
    plt.scatter(X_test[y_test == i, 0], X_test[y_test == i, 1], 
                color=plt.cm.Set1(i),s=100 , marker='x',
                label=f'Actual {target_name}')
    # 绘制预测类别，圆圈代表预测
    plt.scatter(X_test[y_pred == i, 0], X_test[y_pred == i, 1], 
                color=plt.cm.Set1(i), s=100, marker='o', 
                label=f'Predicted {target_name}')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.legend()
plt.title('Classification Scatter Plot')
plt.savefig("../output/Iris_ScatterPlot.png")

# 计算混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
# 可视化混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig("../output/Iris_ConfusionMatrixPlot.png")

'''
手搓一个逻辑回归梯度下降法 学习偏置列的手法
'''

# 加偏置列
X = np.column_stack([np.ones(X.shape[0]), X])   # (150, 5)

# 划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=0)

#one-hot编码标签
Y_train_onehot = np.eye(num_classes)[y_train]   # (90,3)

# ---------- 3. 参数初始化 ----------
n_features = X_train.shape[1]                   # 5
W = np.zeros((n_features, num_classes))         # (5,3)

# ---------- 4. 核心函数 ----------
def softmax(Z):
    """Z: (batch, k)"""
    exp = np.exp(Z - np.max(Z, axis=1, keepdims=True))  # 数值稳定
    return exp / exp.sum(axis=1, keepdims=True)

def cross_entropy_loss(Y_true, Y_prob):
    m = Y_true.shape[0]
    return -np.sum(Y_true * np.log(Y_prob + 1e-15)) / m

# ---------- 5. 训练 ----------
lr = 0.1
epochs = 1000
m = X_train.shape[0]

for epoch in range(epochs):
    # 前向
    logits = X_train @ W          # (90,3)
    probs = softmax(logits)
    loss = cross_entropy_loss(Y_train_onehot, probs)

    # 梯度
    grad = (1 / m) * X_train.T @ (probs - Y_train_onehot)  # (5,3)

    # 更新
    W -= lr * grad

    if epoch % 200 == 0:
        print(f"epoch {epoch:4d}  loss={loss:.4f}")

# ---------- 6. 评估 ----------
probs_test = softmax(X_test @ W)        # (60,3)
y_pred = np.argmax(probs_test, axis=1)
acc = (y_pred == y_test).mean()
print("Test accuracy:", acc)