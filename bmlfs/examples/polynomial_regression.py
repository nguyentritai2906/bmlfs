import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bmlfs.supervised_learning.regression import (PolynomialRegression,
                                                  PolynomialRidgeRegression)
from bmlfs.utils.data_manipulation import k_fold_cross_validation_sets
from bmlfs.utils.data_operation import mean_squared_error, train_test_split

# Hyperparameters
SEED = 42
GRADIENT = False
N_ITERATIONS = 100000
LEARNING_RATE = 0.1
DEGREE = 15
N_FOLD = 5
REGULARIZE = False
REG_RANGE = [0, 0.1, 0.01]

# Import data
dir_name = os.path.dirname(__file__)
data = pd.read_csv(os.path.join(dir_name, '../data/TempLinkoping2016.txt'),
                   sep="\t")

time = np.atleast_2d(data['time'].values).T
temp = data['temp'].values

X = time
y = temp

# Split data
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.4,
                                                    seed=SEED)

# Model
print("Bmlfs")
if REGULARIZE:
    # Finding regularization constant using cross validation
    # TODO: Implement GridSearch
    lowest_mse = float("inf")
    best_reg_factor = None
    print("Finding regularization constant using cross validation")
    for reg_factor in np.arange(*REG_RANGE):
        cross_validation_sets = k_fold_cross_validation_sets(X_train,
                                                             y_train,
                                                             n_fold=N_FOLD)
        mse = 0
        for _X_train, _X_test, _y_train, _y_test in cross_validation_sets:
            model = PolynomialRidgeRegression(degree=DEGREE,
                                              reg_factor=reg_factor,
                                              lr=LEARNING_RATE,
                                              n_iter=N_ITERATIONS)
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
    model = PolynomialRidgeRegression(degree=DEGREE,
                                      reg_factor=best_reg_factor,
                                      lr=LEARNING_RATE,
                                      n_iter=N_ITERATIONS)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Valid \tMSE: %.2f @ Reg: %.2f" % (mse, best_reg_factor))
else:
    model = PolynomialRegression(degree=DEGREE,
                                 lr=LEARNING_RATE,
                                 n_iter=N_ITERATIONS,
                                 gradient=GRADIENT)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Valid \tMSE: %.2f" % mse)

# Predict
y_pred_line = model.predict(X)

# Plot the results
cmap = plt.get_cmap('viridis')  # Color map
m1 = plt.scatter(len(time) * X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(len(time) * X_test, y_test, color=cmap(0.5), s=10)
plt.plot(len(time) * X,
         y_pred_line,
         color='green',
         linewidth=2,
         label="Prediction")
plt.suptitle("Polynomial Ridge Regression")
plt.title("MSE: %.2f" % mse, fontsize=10)
plt.xlabel('Day')
plt.ylabel('Temperature in Celcius')
plt.legend((m1, m2), ("Training data", "Test data"), loc='lower right')
plt.show()

# # Sklearn MSE: 11.36
# import sklearn as sk
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.pipeline import Pipeline

# model = Pipeline([('poly', PolynomialFeatures(degree=DEGREE)),
#                 ('linear', LinearRegression(fit_intercept=False))])
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)
# mse = sk.metrics.mean_squared_error(y_test, y_pred)
# print("Bmlfs")
# print("Valid \tMSE: %.2f" % mse)
