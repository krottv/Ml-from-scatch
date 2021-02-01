import numpy as np
from numpy.lib.arraypad import pad

"""
s = np.array([2, 3, 5, 7, 11])
w = np.array([-1, 1]) 
"""


def convolve(sequence, weights):
    convolution = np.zeros(len(sequence) - len(weights) + 1)
    for i in range(convolution.shape[0]):
        convolution[i] = np.sum(weights * sequence[i:i + len(weights)])
    return convolution 



def convolve2d(sequence, weights):
    convolution = np.zeros((sequence.shape[0] - weights.shape[0] + 1, sequence.shape[1] - weights.shape[1] + 1))

    for i1 in range(convolution.shape[0]):
        for i2 in range(convolution.shape[1]):
            convolution[i1, i2] = np.sum(sequence[i1: i1 + weights.shape[0], i2: i2 + weights.shape[1]] * weights)

    return convolution



"""
s = np.array([[2, 1, 3], 
     [4, 0, 2], 
     [1, 5, 6]])

w = np.array([[-1, -2], 
     [1,   2]])
"""

"""
s1 = np.array([[1, 3, 1], [3, 1, 5]])
s2 = np.array([[2, 2, 3], [3, 2, 1]])
w1 = np.array([[0.5], [0.5]])
w2 = np.array([[1], [-1]])
"""

def calc_new_size(weights, kernels, padding, stride):
    return (np.array(weights.shape[0:-1]) - np.array(kernels.shape[0:-1]) + 2 * padding) / stride + 1

weights = np.zeros([11, 11, 6])
kernels = np.zeros([3, 3, 6])
padding = 4
stride = 2

print(calc_new_size(weights, kernels, padding, stride))

