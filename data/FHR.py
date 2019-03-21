import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler

import pandas as pd

class FHR(object):
    def __init__(self, sampleSize=-1):
        data = pd.read_csv("/home/kaiyuewang/PycharmProjects/unsuperized-learning/data/fetal-hr.csv")
        data.drop(['CLASS'], axis=1)
        list_to_normalize = list(set(data.columns.values) - set(['NSP']))
        y = data['NSP'].values
        X = data[list_to_normalize].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=0)

        std_scale = StandardScaler().fit( self.X_train)
        self.X_train = std_scale.transform( self.X_train)
        self.X_test = std_scale.transform(self.X_test)

    def getTrainingData(self):
        return self.X_train, self.y_train

if __name__=="__main__":
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt

    fhr = FHR()
    trainX = fhr.X_train
    wcss = list()
    rg = range(1, 50)
    for k in rg:
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(trainX)
        wcss.append(kmeans.inertia_)
    plt.figure(figsize=(15, 6))
    plt.plot(rg, wcss, marker="o")
    plt.show()

