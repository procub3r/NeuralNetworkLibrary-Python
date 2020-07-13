import numpy as np


class Layer:
    def __init__(self, length, prev_len=0):
        self.length = length
        self.values = None
        self.weights = ((2 * np.random.rand(length * prev_len)) - 1).reshape((length, prev_len))
        self.biases = np.ones((length, 1))
        self.activations = None
        self.activations_gradient = None

    def activate(self):
        self.activations = self.sigmoid(self.values)

    def d_activate(self, x):
        sig = self.sigmoid(x)
        return sig * (1 - sig)

    def tweak(self, prev_layer, alpha):
        prev_layer.activations_gradient = ((self.activations_gradient * self.d_activate(self.values)).T
                                           .dot(self.weights)).T
        self.weights -= (self.activations_gradient * self.d_activate(self.values)).dot(prev_layer.activations.T) * alpha
        self.biases -= self.activations_gradient * self.d_activate(self.values) * alpha

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))


class NeuralNetwork:
    def __init__(self, dimensions):
        self.dimensions = dimensions
        self.layers = [Layer(dimensions[0])]
        for i in range(1, len(self.dimensions)):
            self.layers.append(Layer(dimensions[i], prev_len=dimensions[i-1]))

    def feedforward(self, inputs):
        self.layers[0].activations = np.array(inputs).reshape((len(inputs), 1))
        for i in range(1, len(self.layers)):
            self.layers[i].values = self.layers[i].weights.dot(self.layers[i-1].activations) + self.layers[i].biases
            self.layers[i].activate()
        return self.layers[-1].activations

    def train(self, input, target, alpha):
        input = np.array(input).reshape((len(input), 1))
        target = np.array(target).reshape((len(target), 1))
        output = self.feedforward(input)
        cost = (output - target) ** 2
        print(np.average(cost))
        self.layers[-1].activations_gradient = 2 * (output - target)
        for i in range(len(self.layers) - 1, 0, -1):
            self.layers[i].tweak(self.layers[i-1], alpha)
