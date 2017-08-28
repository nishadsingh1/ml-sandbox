from mnist import MNIST
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from math import exp, log
import scipy.io
import random
from neural_net import NeuralNet
from utils import *


def pre_process(x):
    return add_column_of_ones(normalize(x))


def report(predictions_validation, predictions_training, labels,
           validation_indices, training_indices):
    def get_frac_correct(indices, preds):
        correct = 0.0
        j = 0
        for i in indices:
            if preds[j] == labels[i]:
                correct += 1
            j += 1
        return correct / float(len(indices))

    training_accuracy = get_frac_correct(training_indices,
                                         predictions_training)
    validation_accuracy = get_frac_correct(validation_indices,
                                           predictions_validation)

    print("Training set accuracy: " + str(100 * training_accuracy) + "%")
    print("Validation set accuracy: " + str(100 * validation_accuracy) + "%")


if __name__ == "__main__":

    SAMPLE_SIZE = 200
    TRAINING_SIZE = 100

    X_train, labels_train, X_test, labels_test = load_dataset()
    training_indices = np.random.choice(
        SAMPLE_SIZE, size=TRAINING_SIZE, replace=False)
    validation_indices = np.setdiff1d(np.arange(SAMPLE_SIZE), training_indices)
    random.shuffle(training_indices)
    data = pre_process(X_train)

    neural_net = NeuralNet()
    iterations, losses, accuracies = neural_net.train(data, labels_train,
                                                      training_indices)

    predictions_validation = neural_net.predict(data, validation_indices)
    predictions_training = neural_net.predict(data, training_indices)

    report(predictions_validation, predictions_training, labels_train,
           validation_indices, training_indices)

    plt.figure()
    plt.plot(iterations, losses)
    plt.title('Training loss by # of iterations')
    plt.figure()
    plt.plot(iterations, accuracies)
    plt.title('Training accuracy by # of iterations')
    plt.show()
