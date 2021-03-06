import numpy as np


class Sigmoid():
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def gradient(self, x):
        return self.__call__(x) * (1 - self.__call__(x))


class Softmax():
    def __call__(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

        # Sklearn implementation
        # max_prob = np.max(X, axis=1).reshape((-1, 1))
        # X -= max_prob
        # np.exp(X, X)
        # sum_prob = np.sum(X, axis=1).reshape((-1, 1))
        # X /= sum_prob
        # return X

    def gradient(self, x):
        p = self.__call__(x)
        return p * (1 - p)
