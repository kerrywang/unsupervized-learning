from sklearn.neural_network import MLPClassifier
from dataTransformer import *
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from data import MNIST
from sklearn.metrics import accuracy_score
from time import time
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection


if __name__=="__main__":
    mnist = MNIST(10000)
    start = time()
    pipeline = Pipeline([('Scale', StandardScaler()), ('PCA', SparseRandomProjection(random_state=0, n_components=160)),
                         ('MLP', MLPClassifier(hidden_layer_sizes=(512, 256), alpha=0.01, verbose=1))])

    pipeline.fit(mnist.X_train, mnist.y_train)
    y_pred = pipeline.predict(mnist.X_test)
    end = time()

    print ("time used: {}s".format(end - start))
    print (accuracy_score(y_pred, mnist.y_test))
# MLPClassifier(hidden_layer_sizes=(512, 256), alpha=0.01)