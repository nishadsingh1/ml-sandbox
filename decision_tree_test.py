from decision_tree import DecisionTree
from utils import *
import random

if __name__ == "__main__":

    # TODO: Find and use a dataset with categorical data
    X_train, labels_train, X_test, labels_test = load_dataset_and_trim()
    print("Loaded dataset")

    model = DecisionTree()
    model.train(X_train, labels_train)
    print("Trained model")

    rand_index = int(random.random() * X_test.shape[0])
    print("Following the path of test datapoint {0}:".format(rand_index))
    model.predict_and_print(X_test[rand_index])

    predictions_train = model.predict(X_train)
    predictions_test = model.predict(X_test)

    training_accuracy = get_frac_equal(predictions_train, labels_train)
    test_accuracy = get_frac_equal(predictions_test, labels_test)

    report_accuracies(training_accuracy, test_accuracy)
