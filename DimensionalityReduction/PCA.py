import matplotlib.pyplot as plt
import os
from util import plot_eigen_value_distribution, scatter_plot_2d
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from matplotlib import cm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from data import MNIST, FHR
import time
import util


class PCAExp(object):
    def __init__(self, X, y):
        self.algo = PCA(random_state=0)
        self.X = X
        self.y = y
        self.algo.fit(self.X)

def PCA_Reconstruction(X, ncomponent):
    mu = np.mean(X, axis=0)

    pca = PCA()
    pca.fit(X)

    trandformedData = pca.transform(X)
    reconstructed = np.dot(trandformedData[:, :ncomponent], pca.components_[:ncomponent, :])
    reconstructed += mu

    errors = np.square(X-reconstructed)
    return reconstructed, np.nanmean(errors)




if __name__=="__main__":
    mnist = MNIST(10000)
    fhr = FHR()
    # errors = []
    # sampleRange= range(1, 41, 1)
    # for k in sampleRange:
    #     reconstruced, error = PCA_Reconstruction(fhr.X_train, k)
    #     errors.append(error)
    # util.plot_regular(sampleRange, errors, "n-component", "MSE", "PCA reconstruction error")

    #
    # reconstructed = PCA_Reconstruction(mnist.X_train, 784)
    #
    # util.plot_Original_Reconstructed(mnist.X_train[0].reshape(28, 28), reconstructed[0].reshape(28, 28))

    start = time.time()
    pca = PCA(random_state=0,n_components=400)
    X = pca.fit_transform(mnist.X_train)
    end = time.time()

    print ("PCA took {}s".format(end - start))
    #
    # scatter_plot_2d(X, fhr.y_train, "PCA 2D plot")
    plot_eigen_value_distribution(pca.explained_variance_)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')

    plt.show()