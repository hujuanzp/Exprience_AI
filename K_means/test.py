import random
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# 查看iris数据集
iris = load_iris()
X = iris.data[:]
print(X)
print(X[:,0])
print(X[:,1])

# plt.scatter(X[:, 0], X[:, 1], c = "red", marker='o', label='see')
# plt.xlabel('petal length')
# plt.ylabel('petal width')
# plt.legend(loc=2)
# plt.show()
