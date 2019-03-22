from sklearn.decomposition import FastICA
import pandas as pd
import util
from data import MNIST, FHR
import matplotlib.pyplot as plt

def getKurt(transformed):
    tmp = pd.DataFrame(transformed)
    tmp = tmp.kurt(axis=0)
    plt.plot(tmp)
    plt.show()
    return tmp.abs().mean()



if __name__ == "__main__":
    fhr = MNIST(1000)
    ica = FastICA(n_components=2, random_state=0, max_iter=500, tol=0.001)
    transformed = ica.fit_transform(fhr.X_train)
    util.scatter_plot_2d(transformed, fhr.y_train, "ICA scattered Plot (MNIST)")

    plt.plot(ica.components_[0], 'r-')
    plt.plot(ica.components_[1], 'g-')
    plt.show()

    # kurt = []
    # for i in range(1, 42):
    #     ica.set_params(n_components=i)
    #     transformed = ica.fit_transform(fhr.X_train)
    #     kurt.append(getKurt(transformed))
    # util.plot_regular(range(1, 42), kurt, "k component", "Kurtosis", "FHR-ICA: Kurtosis vs Number of Component")

