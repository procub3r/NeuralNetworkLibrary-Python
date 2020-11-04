# Demo of the XOR logic gate:

from NeuralNetwork import *

nn = NeuralNetwork([2, 2, 1])

inputs = [[1, 0], [0, 1], [1, 1], [0, 0]]
targets = [[1], [1], [0], [0]]

nn.train(inputs, targets)

print('----------------')

for i in inputs:
	print(f'input: {i}; output: {nn.feedforward(i)}')
