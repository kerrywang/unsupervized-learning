import tensorflow as tf
import numpy as np
from keras.datasets import mnist

class MNIST(object):
    def __init__(self, sampleSize=-1):
        minst = tf.keras.datasets.mnist.load_data(path='mnist.npz')
        (self.X_train, self.y_train), (self.X_test, self.y_test) = minst
        if sampleSize != -1:
            randomSampleIndex = np.random.choice(self.X_train.shape[0], sampleSize)
            self.X_train, self.y_train = self.X_train[randomSampleIndex], self.y_train[randomSampleIndex]

        self.X_train =self.X_train.reshape(self.X_train.shape[0], -1)
        self.X_test = self.X_test.reshape(self.X_test.shape[0], -1)

    def getTrainingData(self):
        return self.X_train, self.y_train

