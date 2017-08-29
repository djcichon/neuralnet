import numpy as np

class ActivationFunction:
    @staticmethod
    def apply(preactivations, activations=None):
        raise Exception("Must implement apply()")

    @staticmethod
    def apply_derivative(preactivations):
        raise Exception("Must implement apply_derivative()")

class Sigmoid(ActivationFunction):
    @staticmethod
    def apply(preactivations, activations=None):
        if activations == None:
            activations = np.empty(preactivations.shape)
        np.multiply(-1, preactivations, out=activations)
        np.exp(activations, out=activations)
        np.add(1, activations, out=activations)
        np.divide(1, activations, out=activations)

        return activations

    @staticmethod
    def apply_derivative(preactivations):
        sig = Sigmoid.apply(preactivations)
        return sig * (1 - sig)

class ReLU(ActivationFunction):
    @staticmethod
    def apply(preactivations, activations=None):
        return np.maximum(0, preactivations)

    @staticmethod
    def apply_derivative(preactivations):
        return (preactivations > 0).astype(float)

