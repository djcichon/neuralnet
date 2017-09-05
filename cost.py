import numpy as np

class CostFunction:
    @staticmethod
    def cost_derivative(actual, expected, preactivations):
        raise Exception("Must implement cost_derivative()")

class QuadraticCost(CostFunction):
    @staticmethod
    def cost_derivative(actual, expected, preactivations):
       return (actual - expected) * sigmoid_prime(preactivations)

class CrossEntropy(CostFunction):
    @staticmethod
    def cost(actual, expected):
        logprobs = np.multiply(expected, np.log(actual)) + np.multiply(1-expected, np.log(1-actual))
        return -np.sum(np.mean(logprobs, axis=1))

    @staticmethod
    def cost_derivative(actual, expected, preactivations):
        return (actual - expected)

