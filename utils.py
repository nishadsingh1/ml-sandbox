import numpy as np
from mnist import MNIST

NUM_CLASSES = 10
DEFAULT_LOCATION  = "/Users/nishadsingh/Documents/Berkeley/cs189/classes/data"

def load_dataset(num_train=200, num_test=10, mnist_location=DEFAULT_LOCATION):
    """Loads in the MNIST dataset
    Download the MNIST dataset from https://pypi.python.org/pypi/python-mnist/0.3
    """
    mndata = MNIST(mnist_location)
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    train_cap = num_train or X_train.shape[0]
    test_cap = num_test or X_test.shape[0]

    return X_train[:train_cap], labels_train[:train_cap], X_test[:test_cap], labels_test[:test_cap]

def one_hot(labels_train, num_classes=NUM_CLASSES):
    """
    Converts categorical labels into standard basis vectors in R^{NUM_CLASSES}
    """
    print(labels_train.shape)
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

relu = np.vectorize(lambda S_hid: max(0, S_hid))
relu_gradient = np.vectorize(lambda S_hid: 1 if S_hid > 0 else 0)
softmax = np.vectorize(lambda S_out, S_out_sum_exp: np.divide(np.exp(S_out), S_out_sum_exp))

def logify(X, logify_add_const):
    return np.log(np.add(X, logify_add_const))

def standardize(x):
    mean = np.mean(x)
    std = np.std(x)
    return np.divide(np.subtract(x, mean), std)