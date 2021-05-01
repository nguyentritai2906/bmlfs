import math
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression

from bmlfs.supervised_learning.regression import PolynomialRegression, PolynomialRidgeRegression
from bmlfs.utils.data_operation import mean_squared_error, train_test_split
from bmlfs.utils.data_manipulation import k_fold_cross_validation_sets


def main():

    # Parse commandline arguments
    parser = argparse.ArgumentParser(
        description='Polynomial Regression model implementation from scratch')
    parser.add_argument("-i", "--iteration", type=int, default=10000, help="Number of training iterations the algorithm will tune the weights for.")
    parser.add_argument("-l", "--lr", type=float, default=0.01, help="Step length that will be used when updating the weights.")
    parser.add_argument("-d", "--degree", type=int, default=15, help="Degree of the polynomial that the independent variance X will be transformed to.")
    parser.add_argument("-f", "--n_fold", type=int, default=5, help="Number of fold to split the training set for cross validation")
    parser.add_argument("-re", "--regularize", type=bool, default=False, help="Whether or not to use regularization")
    parser.add_argument("-rr", "--reg_range", nargs=3, type=float, default=[0, 0.1, 0.01], help="Range to search for best regularization factor [low, high, step]")

    args = parser.parse_args()

    # Hyperparameters
    N_ITERATIONS = args.iteration
    LEARNING_RATE = args.lr
    DEGREE = args.degree
    N_FOLD = args.n_fold
    REGULARIZE = args.regularize
    REG_RANGE = args.reg_range

    data = pd.read_csv('data/TempLinkoping2016.txt', sep="\t")

    time = np.atleast_2d(data['time'].values).T
    temp = data['temp'].values

    X = time
    y = temp

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    if REGULARIZE:
        # Finding regularization constant using cross validation
        # TODO: Implement GridSearch
        lowest_mse = float("inf")
        best_reg_factor = None
        print("Finding regularization constant using cross validation")
        for reg_factor in np.arange(*REG_RANGE):
            cross_validation_sets = k_fold_cross_validation_sets(X_train, y_train, n_fold=N_FOLD)
            mse = 0
            for _X_train, _X_test, _y_train, _y_test in cross_validation_sets:
                model = PolynomialRidgeRegression(degree=DEGREE, reg_factor=reg_factor, learning_rate=LEARNING_RATE, n_iterations=N_ITERATIONS)
                model.fit(_X_train, _y_train)
                y_pred = model.predict(_X_test)
                _mse = mean_squared_error(_y_test, y_pred)
                mse += _mse
            mse /= N_FOLD

            # Print the mean squared error
            print("\tMSE: %.2f @ Reg: %.2f" % (mse, reg_factor))

            # Save regularization factor that gave the lowest error
            if mse < lowest_mse:
                best_reg_factor = reg_factor
                lowest_mse = mse

        # Make final prediction
        model = PolynomialRidgeRegression(degree=DEGREE, reg_factor=best_reg_factor, learning_rate=LEARNING_RATE, n_iterations=N_ITERATIONS)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print("Valid \tMSE: %.2f @ Reg: %.2f" % (mse, best_reg_factor))
    else:
        model = PolynomialRegression(degree=DEGREE, learning_rate=LEARNING_RATE, n_iterations=N_ITERATIONS)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print("Valid \tMSE: %.2f" % mse)

    y_pred_line = model.predict(X)

    # Color map
    cmap = plt.get_cmap('viridis')

    # Plot the results
    m1 = plt.scatter(len(time) * X_train, y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(len(time) * X_test, y_test, color=cmap(0.5), s=10)
    plt.plot(len(time) * X, y_pred_line, color='green', linewidth=2, label="Prediction")
    plt.suptitle("Polynomial Ridge Regression")
    plt.title("MSE: %.2f" % mse, fontsize=10)
    plt.xlabel('Day')
    plt.ylabel('Temperature in Celcius')
    plt.legend((m1, m2), ("Training data", "Test data"), loc='lower right')
    plt.show()


if __name__ == "__main__":
    main()
