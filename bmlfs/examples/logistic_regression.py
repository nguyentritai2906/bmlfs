import numpy as np
import math
import matplotlib.pyplot as plt
from bmlfs.utils import make_diagonal, normalize, train_test_split, accuracy_score
from bmlfs.deep_learning import Sigmoid
from bmlfs.supervised_learning import LogisticRegression
from sklearn import datasets


def main():
    # Load dataset
    iris = datasets.load_iris()
    X = iris.data[:, 3:]  # petal width
    y = (iris.target == 2).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.333, seed=42)

    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: ", accuracy)

    print(clf.predict(np.array([[1.7], [1.5]])))


if __name__ == "__main__":
    main()
