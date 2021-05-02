import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

from bmlfs.supervised_learning.regression import LinearRegression
from bmlfs.utils.data_operation import mean_squared_error, train_test_split

# Hyperparameters
SEED = 42
GRADIENT = True
EPOCH = 300
LEARNING_RATE = 0.01

# Make dataset
X, y = make_regression(n_samples=100,
                       n_features=1,
                       noise=20,
                       random_state=SEED)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.4,
                                                    seed=SEED)

# Define model
model = LinearRegression(n_iter=EPOCH, lr=LEARNING_RATE, gradient=GRADIENT)

model.fit(X_train, y_train)

# Plot training_errors
if GRADIENT:
    # Training error plot
    n = len(model.training_errors)
    training, = plt.plot(range(n),
                         model.training_errors,
                         label="Training Error")
    plt.legend(handles=[training])
    plt.title("Error Plot")
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Iterations')
    plt.show()

# Predict X_test
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Bmlfs Mean squared error: %s" % (mse))

y_pred_line = model.predict(X)

# Plot the results
cmap = plt.get_cmap('viridis')  # Color map
m1 = plt.scatter(366 * X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(366 * X_test, y_test, color=cmap(0.5), s=10)
plt.plot(366 * X, y_pred_line, color='black', linewidth=2, label="Prediction")
plt.suptitle("Linear Regression")
plt.title("MSE: %.2f" % mse, fontsize=10)
plt.xlabel('Day')
plt.ylabel('Temperature in Celcius')
plt.legend((m1, m2), ("Training data", "Test data"), loc='lower right')
plt.show()

# Sklearn MSE: 350.3629
# import sklearn as sk
# from sklearn.linear_model import LinearRegression

# model = LinearRegression()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# mse = sk.metrics.mean_squared_error(y_test, y_pred)
# print("Sklearn Mean squared error: %s" % (mse))
