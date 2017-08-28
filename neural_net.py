import numpy as np
from utils import *
from model import Model

N_IN = 784
N_HID = 5
N_OUT = 10
SIGMA = 0.1
GAMMA = 0.9
INIT_LEARNING_RATE = 0.04
NUM_EPOCHS = 4
ITERATION_REPORT_DELAY = 100

class NeuralNet(Model):
    """
    Implements a Nueral Net with 1 hidden layer.
    The input layer has N_in nodes, the hidden layer has N_hid, and the output layer has N_out.
    """

    def __init__(self, iteration_report_delay=ITERATION_REPORT_DELAY, sigma=SIGMA, init_learning_rate=INIT_LEARNING_RATE, gamma=GAMMA, num_epochs=NUM_EPOCHS, N_in=N_IN, N_hid=N_HID, N_out=N_OUT):
        self.iteration_report_delay = iteration_report_delay
        self.init_learning_rate = INIT_LEARNING_RATE
        self.gamma = gamma
        self.num_epochs = num_epochs

        self.W = sigma * np.random.randn(N_out, N_hid + 1)
        self.V = sigma * np.random.randn(N_hid, N_in + 1)

    def fwd_pass(self, x, V, W):
        S_hid = np.dot(x, V.T)
        H_no_bias = relu(S_hid)
        H = np.append(H_no_bias, 1)
        S_out = np.dot(H, W.T)
        O = softmax(S_out, np.sum(np.exp(S_out)))
        return S_hid, H, O

    def train(self, X_train, Y_train, training_indices):
        learning_rate = self.init_learning_rate
        j = 0
        iterations = []
        losses = []
        accuracies = []
        for _ in range(self.num_epochs):
            for i in training_indices:
                if j % self.iteration_report_delay == 0:
                    iterations.append(j)
                    loss, accuracy = self.document_tl_and_classification_accuracy(X_train, Y_train, self.V, self.W, training_indices)
                    losses.append(loss)
                    accuracies.append(accuracy)
                X = X_train[i]
                Y = Y_train[i]

                S_hid, H, d_out = self.fwd_pass(X, self.V, self.W)

                d_out[Y] -= 1
                dJdW = np.outer(d_out, H)

                delta_hid = np.dot(d_out, self.W)
                S_hid_bias = np.append(S_hid, 0)
                delta_hid = np.dot(delta_hid, np.diag(relu_gradient(S_hid_bias)))
                delta_hid_no_bias = np.delete(delta_hid, len(delta_hid) - 1)
                dJdV = np.outer(delta_hid_no_bias, X)
                
                self.W = np.subtract(self.W, np.multiply(learning_rate, dJdW))
                self.V = np.subtract(self.V, np.multiply(learning_rate, dJdV))
                j += 1
            learning_rate *= self.gamma

    def predict(self, X_processed, validation_indices):
        predictions = []
        for i in validation_indices:
            X = X_processed[i]
            S_hids, H, O = self.fwd_pass(X, self.V, self.W)
            predictions.append(np.argmax(O))
        return np.array(predictions)
