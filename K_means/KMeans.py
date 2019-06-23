# coding=utf-8
from numpy import *
import matplotlib.pyplot as plt

# 加载数据
def loadDataSet(fileName):  # 解析文件，按tab分割字段，得到一个浮点数字类型的矩阵
    dataMat = []              # 文件的最后一个字段是类别标签
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))   # 将每个元素转成float类型,python3和python2的map函数返回值不一样，python3返回迭代器，python2返回list
        dataMat.append(fltLine)
    return dataMat

# 计算欧几里得距离
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2))) # 求两个向量之间的距离

# 构建聚簇中心，取k个(此例中为4)随机质心
def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))   # 每个质心有n个坐标值，总共要k个质心
    for j in range(n):
        minJ = min(dataSet[:,j])
        maxJ = max(dataSet[:,j])
        rangeJ = float(maxJ - minJ)
        centroids[:,j] = minJ + rangeJ * random.rand(k, 1)
    return centroids

# k-means 聚类算法
def kMeans(dataSet, k, distMeans =distEclud, createCent = randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))    # 用于存放该样本属于哪类及质心距离
    # clusterAssment第一列存放该数据所属的中心点，第二列是该数据到中心点的距离
    centroids = createCent(dataSet, k)
    clusterChanged = True   # 用来判断聚类是否已经收敛
    while clusterChanged:
        clusterChanged = False;
        for i in range(m):  # 把每一个数据点划分到离它最近的中心点
            minDist = inf; minIndex = -1;
            for j in range(k):
                distJI = distMeans(centroids[j,:], dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j  # 如果第i个数据点到第j个中心点更近，则将i归属为j
            if clusterAssment[i,0] != minIndex: clusterChanged = True;  # 如果分配发生变化，则需要继续迭代
            clusterAssment[i,:] = minIndex,minDist**2   # 并将第i个数据点的分配情况存入字典
        print(centroids)
        for cent in range(k):   # 重新计算中心点
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A == cent)[0]]   # 去第一列等于cent的所有列
            centroids[cent,:] = mean(ptsInClust, axis = 0)  # 算出这些数据的中心点
    return centroids, clusterAssment
# --------------------测试----------------------------------------------------
# 用测试数据及测试kmeans算法
dataMat = mat(loadDataSet('testSet.txt'))
d = array(dataMat)          ##mat取列还是二维数组
print(d)
print(d[:, 1])

plt.scatter(d[:, 0], d[:, 1], marker='o', label='see')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc=2)
plt.show()

myCentroids, clustAssing = kMeans(dataMat, 5)
print(myCentroids)
print(clustAssing)

x0 = []
x1 = []
x2 = []
x3 = []
x4 = []

for i in range(clustAssing.shape[0]):
    if array(clustAssing)[i][0] == 0:
        x0.append(d[i])
    elif array(clustAssing)[i][0] == 1:
        x1.append(d[i])
    elif array(clustAssing)[i][0] == 2:
        x2.append(d[i])
    elif array(clustAssing)[i][0] == 3:
        x3.append(d[i])
    elif array(clustAssing)[i][0] == 4:
        x4.append(d[i])

plt.scatter(array(x0)[:, 0], array(x0)[:, 1], c = "red", marker='o', label='label0')
plt.scatter(array(x1)[:, 0], array(x1)[:, 1], c = "green", marker='*', label='label1')
plt.scatter(array(x2)[:, 0], array(x2)[:, 1], c = "blue", marker='+', label='label2')
plt.scatter(array(x3)[:, 0], array(x3)[:, 1], c = "black", marker='o', label='label3')
plt.scatter(array(x4)[:, 0], array(x4)[:, 1], c = "pink", marker='o', label='label3')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc=2)
plt.show()
