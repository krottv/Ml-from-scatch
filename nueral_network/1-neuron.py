import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class Neuron:

    def __init__(self, weights, bias = 0):
        self.weights = weights
        self.bias = bias

    def get_output(self, input):

        return sigmoid(input @ self.weights + self.bias)


class NeuralNetwork:

    def __init__(self, weights):

        self.h1 = Neuron(weights)
        self.h2 = Neuron(weights)
        self.o = Neuron(weights)

    def fastforward(self, input):

        output1 = self.h1.get_output(input)
        output2 = self.h2.get_output(input)

        return self.o.get_output(np.array([output1, output2]))



def test():

    n = NeuralNetwork(np.array([0.8, 0.2]))
    output = n.fastforward(np.array([1, 3]))

    print(f'neuron output {output}')

test()