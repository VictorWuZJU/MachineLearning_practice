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

