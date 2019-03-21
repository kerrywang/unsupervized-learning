import matplotlib.pyplot as plt
import matplotlib
from matplotlib import offsetbox
import numpy as np

def scatter_plot_2d(reducedX, y, title):
    plt.scatter(reducedX[:,0], reducedX[:,1], c=y, s=40, cmap='viridis')
    for i, txt in enumerate(y):
        plt.annotate(txt, (reducedX[i, 0], reducedX[i, 1]))
    plt.title(title)
    plt.show()

def plot_eigen_value_distribution(eigenValues, numBin=30):
    result = plt.hist(eigenValues, bins=numBin)
    for i in range(numBin):
        if result[0][i] > 10**-3:
            plt.text(result[1][i], result[0][i], str(result[0][i]))
    plt.title("eigenValue distribution")
    plt.xlabel("bucket")
    plt.ylabel("count")
    plt.show()

def plot_digit(digit_image):
    plt.imshow(digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
    plt.show()

def plot_Original_Reconstructed(original, reconstruced):
    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True)
    axes[0].imshow(original, cmap=matplotlib.cm.binary, interpolation="nearest")
    axes[1].imshow(reconstruced, cmap=matplotlib.cm.binary, interpolation="nearest")
    plt.show()

def plot_regular(x, y, xlabel, ylabel, title):
    plt.plot(x, y, 'o-')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()