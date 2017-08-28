import numpy as np

class ActivationFunction:
    @staticmethod
    def apply(preactivations):
        raise Exception("Must implement apply()")

    @staticmethod
    def apply_derivative(preactivations):
        raise Exception("Must implement apply_derivative()")

class Sigmoid(ActivationFunction):
    @staticmethod
    def apply(preactivations):
        return 1 / (1 + np.exp(-preactivations))

    @staticmethod
    def apply_derivative(preactivations):
        sig = Sigmoid.apply(preactivations)
        return sig * (1 - sig)

class ReLU(ActivationFunction):
    @staticmethod
    def apply(preactivations):
        return np.maximum(0, preactivations)

    @staticmethod
    def apply_derivative(preactivations):
        return (preactivations > 0).astype(float)

