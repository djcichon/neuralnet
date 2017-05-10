
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
    def cost_derivative(actual, expected, preactivations):
       return (actual - expected)

