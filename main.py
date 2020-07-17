from NeuralNetwork import NeuralNetwork
# import matplotlib.pyplot as plt
# import numpy as np
from mlxtend.data import loadlocal_mnist
#
# training, training_targets = loadlocal_mnist(
#     images_path='train-images.idx3-ubyte',
#     labels_path='train-labels.idx1-ubyte'
# )
testing, testing_targets = loadlocal_mnist(
    images_path='t10k-images.idx3-ubyte',
    labels_path='t10k-labels.idx1-ubyte'
)
#
# training_targets = NeuralNetwork.one_hot_enc(training_targets)
testing_targets = NeuralNetwork.one_hot_enc(testing_targets)
#
nn = NeuralNetwork.unpickle('state.obj')
# nn.train(training, training_targets, alpha=0.000001, epochs=1)
# nn.pickle(file='state.obj')
#
nn.test(testing, testing_targets)
