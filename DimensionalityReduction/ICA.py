from sklearn.decomposition import FastICA
import pandas as pd
import util
from data import MNIST, FHR

def getKurt(transformed):
    tmp = pd.DataFrame(transformed)
    tmp = tmp.kurt(axis=0)
    return tmp.abs().mean()



if __name__ == "__main__":
    fhr = FHR()
    ica = FastICA(random_state=0, max_iter=500, tol=0.001)
    kurt = []
    for i in range(1, 42):
        ica.set_params(n_components=i)
        transformed = ica.fit_transform(fhr.X_train)
        kurt.append(getKurt(transformed))
    util.plot_regular(range(1, 42), kurt, "k component", "Kurtosis", "FHR-ICA: Kurtosis vs Number of Component")

