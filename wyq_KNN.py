import numpy as np
import random
import matplotlib
import math
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
from collections import Counter

class dataset:
    def __init__(self, datasetNum, sampleRate, sampleSeed = 114514, testRate = 0.2, testSeed = 114514):
        '''
        datasetNum: 选择数据集(iris/wine/breast_cancel)
        sampleRate: 采样比例
        sampleSeed: 采样随机种子
        testRate:   训练集与测试集比例
        testSeed:   分割数据集种子
        '''
        assert datasetNum in range(0, 2), "Unkown dataset was specified"
        assert sampleRate <= 1 and sampleRate > 0, "sampleRate must between 0~1, not {}".foramt(sampleRate)
        assert testRate <= 1 and sampleRate > 0, "testRate must between 0~1, not {}".format(testRate)

        datasetDict = [datasets.load_iris, datasets.load_wine, datasets.load_breast_cancer]
        #print(datasetDict.__sizeof__)
        datas = datasetDict[datasetNum]()
        #print(datas, type(datas))
        
        random.seed(sampleSeed)
        nSample = math.floor(datas.data.shape[0] * sampleRate)
        idx = random.sample(range(datas.data.shape[0]), nSample)

        self.X = datas.data[idx][:]
        self.Y = datas.target[idx][:]
        self.__counter = Counter(self.Y)
        self.xTrain, self.xTest, self.yTrain, self.yTest = train_test_split(self.X, self.Y, test_size = testRate, random_state = testSeed)
    def statCounter(self):
        print('======================= dataset information =======================')
        print('Total sample number: {}, Feature dimension: {}, Category number: {}'.format(self.X.shape[0], self.X.shape[1], len(self.__counter)))
        for category in self.__counter:
            print('category {} has {} samples'.format(category, self.__counter[category]))


class KNN:
    def __init__(self, data, k = 3, p = 2):
        '''
        data:   dataset类, 包含训练集和测试集
        k:      近邻数, 默认为3
        p:      p范数, 默认为2
        '''
        self.k = k
        self.p = p
        self.data = data
    def predict(self, x):
        dis = []
        for i in range(len(self.data.xTrain)):
            dis.append((abs(np.linalg.norm(self.data.xTrain[i] - x)), self.data.yTrain[i]))
        dis = sorted(dis, key = lambda x:x[0])
        res = []
        for i in range(self.k):
            res.append(dis[i][1])
        categoryCounter = Counter(res)
        mostNeigbor = categoryCounter.most_common(1)[0]
        return mostNeigbor[0]
    def evaluate(self):
        correctPredict = 0
        for i in range(len(self.data.xTest)):
            print(self.predict(self.data.xTest[i]), self.data.yTest[i])
            if self.predict(self.data.xTest[i]) == self.data.yTest[i]:
                correctPredict += 1
        print("With {} test datas, the model's precesion is {:.2%}".format(len(self.data.xTest), correctPredict / len(self.data.xTest)))
        return correctPredict / len(self.data.xTest)


if __name__ == '__main__':
    iris = dataset(0, 0.8, testSeed = 1)
    iris.statCounter()
    irisKNN = KNN(iris)
    irisKNN.evaluate()