import sys
import getopt
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

from supervised_learning.regression import LinearRegression
from utils.regression import mean_squared_error, train_test_split


def main(argv):

    SEED = 42
    GRADIENT = True
    EPOCH = 100
    LEARNING_RATE = 0.01

    try:
        opts, args = getopt.getopt(argv, "hs:g:e:l:", ["seed=", "gradient=", "epoch=", "learning-rate="])
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('linear_regression.py -s <seed> -g <gradient> -e <epoch> -l <learning_rate>')
            sys.exit()
        elif opt in ("-s", "--seed"):
            SEED = int(arg)
        elif opt in ("-g", "--gradient"):
            GRADIENT = eval(arg)
        elif opt in ("-e", "--epoch"):
            EPOCH = int(arg)
        elif opt in ("-l", "--learning-rate"):
            LEARNING_RATE = float(arg)

    X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=SEED)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, seed=SEED)

    model = LinearRegression(n_iterations=EPOCH,
                             learning_rate=LEARNING_RATE,
                             gradient_descent=GRADIENT)

    model.fit(X_train, y_train)

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

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Mean squared error: %s" % (mse))

    y_pred_line = model.predict(X)

    # Color map
    cmap = plt.get_cmap('viridis')
    # Plot the results
    m1 = plt.scatter(366 * X_train, y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(366 * X_test, y_test, color=cmap(0.5), s=10)
    plt.plot(366 * X,
             y_pred_line,
             color='black',
             linewidth=2,
             label="Prediction")
    plt.suptitle("Linear Regression")
    plt.title("MSE: %.2f" % mse, fontsize=10)
    plt.xlabel('Day')
    plt.ylabel('Temperature in Celcius')
    plt.legend((m1, m2), ("Training data", "Test data"), loc='lower right')
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
