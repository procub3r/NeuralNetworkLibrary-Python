from NeuralNetwork import NeuralNetwork
import random

dimensions = [2, 2, 1]
nn = NeuralNetwork(dimensions)
inputs = [[1, 1], [1, 0], [0, 1], [0, 0]]
targets = [[0], [1], [1], [0]]
alpha = 1
order = [i for i in range(len(inputs))]

for epoch in range(1000):
    random.shuffle(order)
    for i in order:
        nn.train(inputs[i], targets[i], alpha)
print('-----------------')
for i in inputs:
    print(nn.feedforward(i))
