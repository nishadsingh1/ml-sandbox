from linear_regression import LinearRegressionModel
from utils import *

if __name__ == "__main__":
    X_train, labels_train, X_test, labels_test = load_dataset_and_trim()
    print("Loaded dataset")

    model = LinearRegressionModel()
    model.train(X_train, labels_train)
    print("Trained model")

    predictions_train = model.predict(X_train)
    predictions_test = model.predict(X_test)

    training_accuracy = get_frac_equal(predictions_train, labels_train)
    test_accuracy = get_frac_equal(predictions_test, labels_test)

    report_accuracies(training_accuracy, test_accuracy)
