from itertools import combinations_with_replacement

import numpy as np


def shuffle_data(X, y, seed=None):
    """ Random shuffle of the samples in X and y """
    if seed:
        np.random.seed(seed)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]


def normalize(X, axis=-1, order=2):
    """ Normalize the dataset X """
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, axis)


def polynomial_features(X, degree):
    """ Make polynomial feature.
    Parameters:
    -----------
    X: ndarray of shape (n_samples, n_features)
        The input samples
    degree: int
        The highest of the degrees of the polynomial's monomials (individual terms) with non-zero coefficients
    Returns:
    -----------
    X_new: ndarray of shape (n_samples, n_output_features)
        New feature matrix consisting of all polynomial combinations of the features with degree less than or equal to the specified degree. For example, if an input sample is two dimensional and of the form [a, b], the degree-2 polynomial features are [1, a, b, a^2, ab, b^2].

    """
    n_samples, n_features = np.shape(X)

    def index_combinations():
        combs = [
            combinations_with_replacement(range(n_features), i)
            for i in range(0, degree + 1)
        ]
        flat_combs = [item for sublist in combs for item in sublist]
        return flat_combs

    combinations = index_combinations()
    n_output_features = len(combinations)
    X_new = np.empty((n_samples, n_output_features))

    for i, index_combs in enumerate(combinations):
        X_new[:, i] = np.prod(X[:, index_combs], axis=1)

    return X_new


def k_fold_cross_validation_sets(X, y, n_fold, shuffle=True, seed=None):
    """ Split the data into train and test sets """
    if shuffle:
        X, y = shuffle_data(X, y, seed)

    n_samples = len(y)
    left_overs = {}
    n_left_overs = n_samples % n_fold
    if n_left_overs != 0:
        left_overs["X"] = X[-n_left_overs:]
        left_overs["y"] = y[-n_left_overs:]
        X = X[:-n_left_overs]
        y = y[:-n_left_overs]

    X_split = np.split(X, n_fold)
    y_split = np.split(y, n_fold)
    sets = []
    for i in range(n_fold):
        # Use the i-th fold for testing
        X_test, y_test = X_split[i], y_split[i]
        # Train on the rest
        X_train = np.concatenate(X_split[:i] + X_split[i + 1:], axis=0)
        y_train = np.concatenate(y_split[:i] + y_split[i + 1:], axis=0)
        sets.append([X_train, X_test, y_train, y_test])

    # Add left over samples to last set as training samples
    if n_left_overs != 0:
        np.append(sets[-1][0], left_overs["X"], axis=0)
        np.append(sets[-1][2], left_overs["y"], axis=0)

    return np.array(sets, dtype=object)


def train_test_split(X, y, test_size=0.5, shuffle=True, seed=None):
    """ Split the data into train and test sets """
    if shuffle:
        X, y = shuffle_data(X, y, seed)
    # Split the training data from test data in the ratio specified in
    # test_size
    split_i = int(len(y) * (1 - test_size))
    X_train, X_test = X[:split_i], X[split_i:]
    y_train, y_test = y[:split_i], y[split_i:]

    return X_train, X_test, y_train, y_test


def mean_squared_error(y_true, y_pred):
    """ Returns the mean squared error between y_pred and y_true """
    mse = np.mean((y_true - y_pred)**2)
    return mse
