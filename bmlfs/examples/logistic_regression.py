import numpy as np
from sklearn import datasets

from bmlfs.supervised_learning import LogisticRegression
from bmlfs.utils import accuracy_score, train_test_split

# Hyperparameters
SEED = 42

# Load dataset
iris = datasets.load_iris()
X = iris.data[:, 3:]  # petal width
y = (iris.target == 2).astype(int)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.333,
                                                    seed=SEED)

# Define model
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Bmlfs Accuracy: ", accuracy)

# Predict [1, 0]
X_new = np.array([[1.7], [1.5]])
print("Bmlfs Predict: ", model.predict(X_new))
print("Bmlfs Predict probability: ")
print(model.predict_proba(X_new))

# # Sklearn Accuracy: 0.98
# import sklearn as sk
# from sklearn.linear_model import LogisticRegression

# model = LogisticRegression(random_state=SEED)
# model.fit(X_train,y_train)
# y_pred = model.predict(X_test)
# print("\nSklearn Accuracy: ", sk.metrics.accuracy_score(y_test, y_pred))
# print("Sklearn Predict: ", model.predict(X_new))
# print("Sklearn Predict probability: ")
# print(model.predict_proba(X_new))
