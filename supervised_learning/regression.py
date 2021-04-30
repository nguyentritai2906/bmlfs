import math

import numpy as np

from utils.data_manipulation import normalize, polynomial_features


class l1_regulation():
    """ Regularization for Lasso Regression """

    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, w):
        return self.alpha * np.linalg.norm(w, 1)  # np.max(np.sum(np.abs(w), axis=0))

    def grad(self, w):
        return self.alpha * np.sign(w)


class l2_regularization():
    """ Regularization for Ridge Regression """

    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, w):
        return self.alpha * 0.5 * w.T @ w

    def grad(self, w):
        return self.alpha * w


class Regression(object):
    """ Base regression model. Models the relationship between a scalar dependent variable y and the independent variable X.
    Parameters:
    -----------
    n_iterations: int
        The number of training iterations the algorithm will tune the weights for.
    learning_rate: float
        The step length that will be used when updating the weights.
    """

    def __init__(self, n_iterations=100, learning_rate=0.001):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate

    def initialize_weights(self, n_features):
        """ Initialize weights randomly using Xavier Initialization method [-1/N, 1/N] """
        limit = 1 / math.sqrt(n_features)
        self.w = np.random.uniform(-limit, limit, (n_features, ))

    def fit(self, X, y):
        # Insert constant ones for bias weights
        X_b = np.insert(X, 0, 1, axis=1)
        self.training_errors = []
        self.initialize_weights(n_features=X_b.shape[1])
        self.m, self.n = X_b.shape

        # Do gradient descent for n_iterations
        for _ in range(self.n_iterations):
            y_pred = X_b @ self.w
            # Calculate l2 loss
            mse = np.mean(0.5 * (y - y_pred)**2)
            self.training_errors.append(mse)
            # Gradient of l2 loss w.r.t w
            grad_w = 2 / self.m * X_b.T @ (y_pred - y)
            # Update the weights
            self.w -= self.learning_rate * grad_w

    def predict(self, X):
        # Insert constant ones for bias weights
        X_b = np.insert(X, 0, 1, axis=1)
        y_pred = X_b @ self.w
        return y_pred


class LinearRegression(Regression):
    """ Linear regression model.
    Parameters:
    -----------
    n_iterations: int
        The number of training iterations the algorithm will tune the weights for.
    learning_rate: float
        The step length that will be used when updating the weights.
    gradient_descent: boolean
        True or false depending if gradient descent should be used when training. If false then we use batch optimization by least squares.
    """

    def __init__(self,
                 n_iterations=100,
                 learning_rate=0.001,
                 gradient_descent=True):
        self.gradient_descent = gradient_descent
        self.regularization = lambda _: 0
        self.regularization.grad = lambda _: 0
        super(LinearRegression, self).__init__(n_iterations=n_iterations,
                                               learning_rate=learning_rate)

    def fit(self, X, y):
        # If not gradient_descent => Least squares approximation of w
        # https://www.youtube.com/watch?v=ZUU57Q3CFOU - Four Ways to Solve Least Squares Problems by Prof. William Gilbert Strang
        # https://www.youtube.com/watch?v=PjeOmOz9jSY&t=139s - Linear Systems of Equations, Least Squares Regression, Pseudoinverse by Steve Brunton
        # https://www.youtube.com/watch?v=02QCtHM1qb4 - Least Squares Regression and the SVD by Steve Brunton
        if not self.gradient_descent:
            X_b = np.insert(X, 0, 1, axis=1)
            # Calculate weights by least squares (using Moore-Penrose pseudoinverse)
            # X @ w = y --> w = X_pin @ y
            # X_pinv = inv(X.T @ X) @ X.T
            # Normal Equations: w = inv(X.T @ X) @ X.T @ y
            X_pin = np.linalg.pinv(X_b)
            self.w = X_pin @ y
        else:
            super(LinearRegression, self).fit(X, y)


class PolynomialRegression(Regression):
    """Performs a non-linear transformation of the data before fitting the model
    and doing predictions which allows for doing non-linear regression.
    Parameters:
    -----------
    degree: int
        The degree of the polynomial that the independent variable X will be transformed to.
    n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    learning_rate: float
        The step length that will be used when updating the weights.
    """

    def __init__(self, degree, n_iterations=3000, learning_rate=0.001):
        self.degree = degree
        # No regularization
        self.regularization = lambda x: 0
        self.regularization.grad = lambda x: 0
        super(PolynomialRegression, self).__init__(n_iterations=n_iterations,
                                                   learning_rate=learning_rate)

    def fit(self, X, y):
        X = polynomial_features(X, degree=self.degree)
        super(PolynomialRegression, self).fit(X, y)

    def predict(self, X):
        X = polynomial_features(X, degree=self.degree)
        return super(PolynomialRegression, self).predict(X)


class RidgeRegression(Regression):
    """ Also referred to as Tikhonov regularization. Linear regression model with a regularization factor.
    Model that tries to balance the fit of the model with respect to the training data and the complexity of the model. A large regularization factor with decreases the variance of the model.
    Parameters:
    -----------
    reg_factor: float
        The factor that will determine the amount of regularization and feature shrinkage.
    n_iterations: int
        The number of training iterations the algorithm will tune the weights for.
    learning_rate: float
        The step length that will be used when updating the weights.
    """

    def __init__(self, reg_factor, n_iterations=1000, learning_rate=0.001):
        self.regularization = l2_regularization(alpha=reg_factor)
        super(RidgeRegression, self).__init__(n_iterations, learning_rate)


class PolynomialRidgeRegression(Regression):
    """ Similar to regular Ridge Regression except that the data is transformed to allow for polynomial regression.
    Parameters:
    -----------
    degree: int
        The degree of the polynomial that the independent variance X will be transformed to.
    reg_factor: float
        The factor that will determine the amount of regularization and feature shrinkage.
    n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    learning_rate: float
        The step length that will be used when updating the weights.
    """

    def __init__(self,
                 degree,
                 reg_factor,
                 n_iterations=3000,
                 learning_rate=0.01,
                 gradient_descent=True):
        self.degree = degree
        self.regularization = l2_regularization(alpha=reg_factor)
        super(PolynomialRidgeRegression,
              self).__init__(n_iterations, learning_rate)

    def fit(self, X, y):
        X = normalize(polynomial_features(X, degree=self.degree))
        super(PolynomialRidgeRegression, self).fit(X, y)

    def predict(self, X):
        X = normalize(polynomial_features(X, degree=self.degree))
        return super(PolynomialRidgeRegression, self).predict(X)