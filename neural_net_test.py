from neural_net import NeuralNet
from utils import *


def pre_process(x):
    return add_column_of_ones(normalize(x))


if __name__ == "__main__":

    X_train, labels_train, X_test, labels_test = load_dataset_and_trim()
    print("Loaded dataset")

    train_data = pre_process(X_train)
    test_data = pre_process(X_test)
    print("Pre-processed data")

    neural_net = NeuralNet()
    neural_net.train(train_data, labels_train)
    print("Trained model")

    predictions_train = neural_net.predict(train_data)
    predictions_test = neural_net.predict(test_data)

    training_accuracy = get_frac_equal(predictions_train, labels_train)
    test_accuracy = get_frac_equal(predictions_test, labels_test)

    report_accuracies(training_accuracy, test_accuracy)
