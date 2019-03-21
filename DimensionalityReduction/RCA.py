from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
import numpy as np
from data import MNIST, FHR
import util
from scipy.linalg import pinv
import scipy.sparse as sps
from sklearn.metrics.pairwise import pairwise_distances
import time

def RCA_Reconstruction(X, ncomponent):

    start = time.time()
    rca = SparseRandomProjection(random_state=0, n_components=ncomponent)
    rca.fit(X)
    end = time.time()
    print ("RCA took {} s".format(end - start))
    w = rca.components_
    if sps.issparse(w):
        w = w.todense()
    p = pinv(w)
    reconstructed = ((p @ w) @ (X.T)).T  # Unproject projected data
    errors = np.square(X - reconstructed)
    return reconstructed, np.nanmean(errors)

def pairwise_dist_corr(x1, x2):
    assert x1.shape[0] == x2.shape[0]

    d1 = pairwise_distances(x1)
    d2 = pairwise_distances(x2)
    return np.corrcoef(d1.ravel(), d2.ravel())[0, 1]


if __name__ == "__main__":
    mnist = MNIST(10000)
    fhr = FHR()
    #
    restart = 100
    corr = []
    for i in range(restart):
        rca = SparseRandomProjection(n_components=120)
        corr.append(pairwise_dist_corr(rca.fit_transform(mnist.X_train), mnist.X_train))
        print (i)
    util.plot_regular(range(restart), corr, "restart", "pairwise dist corr", "RCA restart tests")
    print (np.var(corr))
    #
    # reconstructed, error = RCA_Reconstruction(mnist.X_train, 400)
    #
    # util.plot_Original_Reconstructed(mnist.X_train[0].reshape(28, 28), reconstructed[0].reshape(28, 28))
    sampleRange= range(1, 42, 1)

    corr = []
    for k in sampleRange:
        rca = SparseRandomProjection(n_components=k)
        corr.append(pairwise_dist_corr(rca.fit_transform(mnist.X_train), mnist.X_train))
    util.plot_regular(sampleRange, corr, "Number of Components", "pairwise dist corr", "FHR - RP Pairwise distance corrcoef vs Number of Components")

    errors = []
    for k in sampleRange:
        reconstruced, error = RCA_Reconstruction(fhr.X_train, k)
        errors.append(error)
        print (k)
    util.plot_regular(sampleRange, errors, "n-component", "MSE", "RCA reconstruction error")