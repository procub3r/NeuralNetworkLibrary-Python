from NeuralNetwork import NeuralNetwork

nn = NeuralNetwork([2, 2, 1])
inputs = [[1, 1], [1, 0], [0, 1], [0, 0]]
targets = [[0], [1], [1], [0]]
nn.train(inputs, targets, alpha=1, epochs=2000)

print('-----------------')

for i in inputs:
    print(nn.feedforward(i))
