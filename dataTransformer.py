import abc
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler


class DataTransform(abc.ABC):
    def __init__(self):
        self.random_state = 0
        self.algo = None

    def fit(self, X, y):
        '''
        transform input data in to processed data
        :param X: features
        :param y: classification
        :return: transformed X and y
        '''
        return self.algo.fit_transform(X), y


    def transform(self, testX, testY):
        '''
        transform input data based on previously fitted data
        :param testX:
        :param testY:
        :return: transformed testX and testY
        '''
        return self.algo.transform(testX), testY

class ZeroMean(DataTransform):
    def __init__(self):
        super.__init__()
        self.algo = StandardScaler()


class PCATransform(DataTransform):
    def __init__(self, n_component=None):
        super().__init__()
        self.algo = PCA(random_state=self.random_state, n_components=n_component)


class ICATransform(DataTransform):
    def __init__(self,  n_component=None):
        self.algo = FastICA(random_state=self.random_state, n_components=n_component)



class DataTransformApply(object):
    def __init__(self, listOfTransform):
        self.listOfTransform = listOfTransform

    def transformData(self, X, y):
        for transformAlgo in self.listOfTransform:
            assert isinstance(transformAlgo, DataTransform)
            X, y = transformAlgo.fit(X, y)
        return X, y

    def evalTestData(self, testX, testy):
        for transformAlgo in self.listOfTransform:
            assert isinstance(transformAlgo, DataTransform)
            testX, testy = transformAlgo.transform(testX, testy)
        return testX, testy


