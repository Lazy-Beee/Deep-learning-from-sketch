import sys
import os
from common.functions import *
sys.path.append(os.pardir)


class TwoLayerNet:
    """A two layer neural net for stochastic gradient decent practice"""

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        """Initialize parameters"""
        self.parameters = {
            'W1': weight_init_std * np.random.randn(input_size, hidden_size),
            'b1': np.zeros(hidden_size),
            'W2': weight_init_std * np.random.randn(hidden_size, output_size),
            'b2': np.zeros(output_size)
        }

    def predict(self, x):
        """Prediction of neural net"""
        W1, W2 = self.parameters['W1'], self.parameters['W2']
        b1, b2 = self.parameters['b1'], self.parameters['b2']
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        return y

    def loss(self, x, t):
        """Loss function of neural net"""
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        """Calcualte accuracy of prediction"""
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        """CAlculate gradient of weights"""
        loss_W = lambda W: self.loss(x, t)
        grads = {
            'W1': numerical_gradient(loss_W, self.parameters['W1']),
            'b1': numerical_gradient(loss_W, self.parameters['b1']),
            'W2': numerical_gradient(loss_W, self.parameters['W2']),
            'b2': numerical_gradient(loss_W, self.parameters['b2'])
        }
        return grads