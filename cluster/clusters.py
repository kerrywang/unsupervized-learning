import abc
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import time
import collections
from data import MNIST, FHR
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection


class AbstactCluster(abc.ABC):
    def __init__(self, X, y, k):
        self.X = X
        self.y = y
        self.nClusters = k
        self.algo = None


    @abc.abstractmethod
    def performCluster(self):
        pass

    @abc.abstractmethod
    def prediction(self):
        pass

    def getClassification(self):
        belongedCentroid = self.prediction()
        centroidDict = collections.defaultdict(list)
        self.centroidLabel = collections.defaultdict(int)
        for i in range(self.X.shape[0]):
            centroidDict[belongedCentroid[i]].append(self.y[i])
        # assign each centroid with maximum number of label
        for centroid, labelSet in centroidDict.items():
            self.centroidLabel[centroid] = max(set(labelSet), key=labelSet.count)
        predictedLabel = list(map(lambda x: self.centroidLabel[x], self.algo.predict(self.X)))
        return accuracy_score(self.y, predictedLabel)

class KMeanCluster(AbstactCluster):
    def __init__(self, X, y, k):
        super().__init__(X, y, k)
        self.algo = KMeans(n_clusters=self.nClusters, random_state=0)

    def performCluster(self):
        startTime = time.time()
        self.algo.fit(self.X, self.y)
        endTime = time.time()
        print("Finished Kmean with {} clusters using {}s".format(self.nClusters, endTime - startTime))
        return sum(np.min(cdist(self.X, self.algo.cluster_centers_, 'euclidean'), axis=1)) / self.X.shape[0]

    def getBiasScore(self):
        return sum(np.min(cdist(self.X, self.algo.cluster_centers_, 'euclidean'), axis=1)) / self.X.shape[0]


    def prediction(self):
        return self.algo.predict(self.X)


class ExpectedMaximization(AbstactCluster):
    def __init__(self, X, y, k):
        super().__init__(X, y, k)
        self.algo = GaussianMixture(n_components=self.nClusters, random_state=0)

    def performCluster(self):
        startTime = time.time()
        self.algo.fit(self.X)
        endTime = time.time()
        print("Finished ExpectedMaximization with {} clusters using {}s".format(self.nClusters, endTime - startTime))


    def getBiasScore(self):
        distance = cdist(self.X, self.algo.means_, 'euclidean')
        return sum(np.sum(np.multiply(cdist(self.X, self.algo.means_, 'euclidean'), self.algo.predict_proba(self.X)), axis=1)) / self.X.shape[0]

    def prediction(self):
        return self.algo.predict(self.X)



if __name__=="__main__":
    # mnist = MNIST(10000)
    fhr = FHR()
    mnist = MNIST(10000)
    print (fhr.X_train.shape)

    trainX = FastICA(random_state=0, n_components=160).fit_transform(mnist.X_train)
    trainY = mnist.y_train
    # trainX = FHR.X_train
    # trainY = FHR.y_train

    scores = []
    compareScores = []
    kRange = range(2, 41, 1)
    for k in kRange:
        newCluster = ExpectedMaximization(trainX, trainY, k)
        newCluster.performCluster()
        testCluster = ExpectedMaximization(mnist.X_train, mnist.y_train, k)
        testCluster.performCluster()

        compareScores.append(testCluster.getClassification())
        scores.append(newCluster.getClassification())
    plt.plot(kRange, scores, 'o-')
    plt.plot(kRange, compareScores, 'ro-')
    plt.legend(['RCA-processed', 'Original'])
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.title("RCA + EM cluster Form MNIST (Accuracy)")
    plt.show()


    # newCluster = ExpectedMaximization(mnist.X_train, mnist.y_train, 10)
    # newCluster.performCluster()
    # print(newCluster.getClassification())


    #
    # scores = []
    # kRange = range(1, 20)
    # for k in kRange:
    #     newCluster = ExpectedMaximization(fhr.X_train, fhr.y_train, k)
    #     newCluster.performCluster()
    #     scores.append(newCluster.getBiasScore())
    #
    # plt.plot(kRange, scores, 'o-')
    # plt.xlabel('k')
    # plt.ylabel('bias')
    # plt.show()