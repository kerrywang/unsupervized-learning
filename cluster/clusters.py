import abc
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import time
import collections
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

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
        self.algo = GaussianMixture(n_components=self.nClusters)

    def performCluster(self):
        startTime = time.time()
        self.algo.fit(self.X)
        endTime = time.time()
        print("Finished ExpectedMaximization with {} clusters using {}s".format(self.nClusters, endTime - startTime))


    def getBiasScore(self):
        return self.algo.bic(self.X)

    def prediction(self):
        return np.argmax(self.algo.predict(self.X), axis=1)



