# coding:utf-8

import random
from sklearn.datasets import load_iris
from sklearn import neighbors
from sklearn import metrics

# 查看iris数据集
iris = load_iris()

knn = neighbors.KNeighborsClassifier(n_neighbors = 3)

trainingSet = []
trainingTarget = []
testSet = []
testTarget = []

for x in range(0, len(iris.data)-1):
    if random.random() < 0.67:
        trainingSet.append(iris.data[x])
        trainingTarget.append(iris.target[x])
    else:
        testSet.append(iris.data[x])
        testTarget.append(iris.target[x])

knn.fit(trainingSet, trainingTarget)
y_predict = knn.predict(testSet)
accuracy_score = metrics.accuracy_score(testTarget, y_predict)
print('accuracy score: %f' % (accuracy_score*100,))
y_predict = knn.predict_proba(testSet)
log_loss = metrics.log_loss(testTarget, y_predict)
print('log loss: %f' % (log_loss,))

