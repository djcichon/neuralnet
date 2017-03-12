import numpy as np

def sigmoid(preactivations):
    return 1 / (1 + np.exp(-preactivations))

def sigmoid_prime(preactivations):
    sig = sigmoid(preactivations)
    return sig * (1 - sig)

