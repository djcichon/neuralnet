import numpy as np
import math

class Layer:

    def __init__(self, size):
        self.next = None
        self.prev = None
        self.biases = None
        self.weights = None
        self.bias_gradient = None
        self.weight_gradient = None

        self.size = size
        self.activations = np.zeros((size, 1))
        self.preactivations = np.zeros((size, 1))
        self.errors = np.zeros((size, 1))

    def connect_to_neighbors(self, prev_layer, next_layer):
        self.prev = prev_layer
        self.next = next_layer

        # Weights & biases should only be defined after the previous layer is known
        if prev_layer:
            self.weights = np.random.randn(self.size, prev_layer.size)/math.sqrt(self.size * prev_layer.size)
            self.weight_gradient = np.empty(self.weights.shape)

            self.biases = np.random.randn(self.size, 1) / math.sqrt(self.size)
            self.bias_gradient = np.empty(self.biases.shape)

    # Populates the next layer using this layer's activations, weights, and biases
    def forward(self):
        self.next.preactivations = np.dot(self.next.weights, self.activations) + self.next.biases
        self.next.activations = sigmoid(self.next.preactivations)

    def calculate_errors(self, expected_outputs, cost_derivative):
        if self.next == None:
            # Output layer just uses the derivative of the cost function
            self.errors = cost_derivative(self.activations, expected_outputs, self.preactivations)
        else:
            # The hidden layers feed errors backwards
            self.errors = np.dot(self.next.weights.T, self.next.errors) * sigmoid_prime(self.preactivations)

    def reset_gradients(self):
        self.bias_gradient.fill(0)
        self.weight_gradient.fill(0)

    def calculate_bias_gradient(self):
        return self.errors
    
    def calculate_weight_gradient(self):
        return np.dot(self.errors, self.prev.activations.T)

#TODO: These need to only be in one place, do I need a util file?
def sigmoid(preactivations):
    return 1 / (1 + np.exp(-preactivations))

def sigmoid_prime(preactivations):
    sig = sigmoid(preactivations)
    return sig * (1 - sig)

