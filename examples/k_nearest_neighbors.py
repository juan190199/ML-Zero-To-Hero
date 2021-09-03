import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets

from supervised_learning.k_nearest_neighbors import KNN

from utils.data_manipulation import train_test_split, normalize
from utils.data_operation import accuracy_score
from utils.misc import Plot


def main():
    data = datasets.load_iris()
    X = normalize(data.data)
    y = data.target
    C = len(np.unique(y))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    clf = KNN(k=5, C=C)
    y_pred = clf.predict(X_test, X_train, y_train)

    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy: ", accuracy)

    # Reduce dimensions to 2d using pca and plot the results
    Plot().plot_in_2d(X_test, y_pred, title="K Nearest Neighbors", accuracy=accuracy, legend_labels=data.target_names)


if __name__ == "__main__":
    main()
