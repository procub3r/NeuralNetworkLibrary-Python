import numpy as np
import random
import pickle


class Layer:
    def __init__(self, length, prev_len=0):
        self.length = length
        self.values = None
        self.weights = ((2 * np.random.rand(length * prev_len)) - 1).reshape((length, prev_len))
        self.biases = np.ones((length, 1))
        self.activations = None
        self.activations_gradient = None

    def activate(self):
        self.activations = Layer.sigmoid(self.values)

    def d_activate(self, x):
        sig = self.sigmoid(x)
        return sig * (1 - sig)

    def tweak(self, prev_layer, alpha):
        prev_layer.activations_gradient = self.weights.T.dot(self.activations_gradient * self.d_activate(self.values))
        self.weights -= ((self.activations_gradient * self.d_activate(self.values)).dot(prev_layer.activations.T)) * alpha
        self.biases -= (self.activations_gradient * self.d_activate(self.values)) * alpha

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

    def train(self, inputs, targets, alpha=1, epochs=1000):
        order = [i for i in range(len(inputs))]
        for i in range(epochs):
            random.shuffle(order)
            for indx, j in enumerate(order):
                input = inputs[j]
                target = targets[j]
                output = self.feedforward(input)
                cost = (output - target) ** 2
                print(i, indx, np.average(cost))
                self.layers[-1].activations_gradient = 2 * (output - target)
                for k in range(len(self.layers) - 1, 0, -1):
                    self.layers[k].tweak(self.layers[k-1], alpha)
                    
    def test(self, test_data, test_labels):
        correct = 0
        for i in range(len(test_data)):
            output = self.feedforward(test_data[i])
            predicted = np.where(np.isclose(output, max(output)))
            answer = np.where(np.isclose(test_labels[i], max(test_labels[i])))
            if predicted == answer:
                correct += 1
            print(f'Tested: {i} / {len(test_data)}', end='\r')
        print(f'Accuracy: {(correct / len(test_data)) * 100}% --- Correct: {correct} / {len(test_data)}')

    @staticmethod
    def one_hot_enc(values, val_range=10):
        result = np.zeros((len(values), val_range, 1))
        for i in range(len(values)):
            result[i][values[i]][0] = 1
        return result

    def pickle(self, file='state.obj'):
        with open(file, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def unpickle(file='state.obj'):
        with open(file, 'rb') as f:
            return pickle.load(f)
