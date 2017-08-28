import sklearn.metrics as metrics
from linear_regression import LinearRegressionModel
from utils import *

if __name__ == "__main__":
    X_train, labels_train, X_test, labels_test = load_dataset()
    print("Loaded dataset")

    model = LinearRegressionModel()
    model.train(X_train, labels_train)
    print("Trained model")

    pred_labels_train = model.predict(X_train)
    pred_labels_test = model.predict(X_test)

    print("Train accuracy: {0}".format(metrics.accuracy_score(labels_train, pred_labels_train)))
    print("Test accuracy: {0}".format(metrics.accuracy_score(labels_test, pred_labels_test)))
