import numpy as np
from mnist import MNIST

NUM_CLASSES = 10
NUM_TRAIN = 200
NUM_TEST = 100
DEFAULT_LOCATION = "/example/path"


def load_dataset(mnist_location=DEFAULT_LOCATION):
    """Loads in the MNIST dataset
    Download the MNIST dataset from https://pypi.python.org/pypi/python-mnist/0.3
    """
    mndata = MNIST(mnist_location)
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())

    return X_train, labels_train, X_test, labels_test


def trim(X, Y, num_datapoints):
    assert (X.shape[0] >= num_datapoints and Y.shape[0] >= num_datapoints)
    return X[:num_datapoints], Y[:num_datapoints]


def load_dataset_and_trim(num_train=NUM_TRAIN,
                          num_test=NUM_TEST,
                          mnist_location=DEFAULT_LOCATION):
    X_train, labels_train, X_test, labels_test = load_dataset(mnist_location)

    X_train_trim, labels_train_trim = trim(X_train, labels_train, num_train)
    X_test_trim, labels_test_trim = trim(X_test, labels_test, num_test)

    return X_train_trim, labels_train_trim, X_test_trim, labels_test_trim


def one_hot(labels_train, num_classes=NUM_CLASSES):
    """
    Converts categorical labels into standard basis vectors in R^{NUM_CLASSES}
    """
    n = labels_train.shape[0]
    basis_vectors = np.zeros((n, NUM_CLASSES))
    for i, label in enumerate(labels_train):
        basis_vectors[i][label] = 1
    return basis_vectors


def normalize(X):
    max_val = X.max()
    return np.divide(np.subtract(X, np.mean(X, axis=0)), max_val)


def add_column_of_ones(x):
    m, n = x.shape
    column_of_ones = np.ones(m)
    return np.column_stack((x, column_of_ones))


def get_frac_equal(preds, labels):
    assert (len(preds) == len(labels))
    correct = 0.0
    for pred, label in zip(preds, labels):
        correct += int(pred == label)
    return correct / float(len(preds))


def report_accuracies(training_accuracy, test_accuracy):
    print("Training set accuracy: {0}%".format(str(100 * training_accuracy)))
    print("Test set accuracy: {0}%".format(str(100 * test_accuracy)))


relu = np.vectorize(lambda S_hid: max(0, S_hid))
relu_gradient = np.vectorize(lambda S_hid: 1 if S_hid > 0 else 0)
softmax = np.vectorize(
    lambda S_out, S_out_sum_exp: np.divide(np.exp(S_out), S_out_sum_exp))
