from sklearn import datasets
from bmlfs.unsupervised_learning import KMeans
from bmlfs.utils import Plot


def main():
    X, y = datasets.make_blobs()

    classifier = KMeans(k=3)
    y_pred = classifier.predict(X)

    p = Plot()
    p.plot_in_2d(X, y_pred, title="K-Means Clustering")
    p.plot_in_2d(X, y, title="Groundtruth Clustering")


if __name__ == "__main__":
    main()
