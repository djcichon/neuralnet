import numpy as np
import math
from activation import *

class Layer:

    def __init__(self, size, activation_function=Sigmoid, dropout_keep_prob=1.):
        self.next = None
        self.prev = None
        self.biases = None
        self.weights = None
        self.bias_gradient = None
        self.weight_gradient = None

        self.size = size
        self.activation_function = activation_function
        self.dropout_keep_prob = dropout_keep_prob

    def connect_to_previous_layer(self, prev_layer):
        self.prev = prev_layer

        # Weights & biases should only be defined after the previous layer is known
        if prev_layer:
            self.prev.next = self
            
            self.weights = np.random.randn(self.size, prev_layer.size)/math.sqrt(self.size * prev_layer.size)
            self.weight_gradient = np.empty(self.weights.shape)

            self.biases = np.random.randn(self.size, 1) / math.sqrt(self.size)
            self.bias_gradient = np.empty(self.biases.shape)

    def initialize_matrices(self, batch_size):
        self.activations = np.zeros((self.size, batch_size))
        self.preactivations = np.zeros((self.size, batch_size))
        self.errors = np.zeros((self.size, batch_size))

    # Populates the next layer using this layer's activations, weights, and biases
    def forward(self):
        np.add(np.dot(self.next.weights, self.activations, self.next.preactivations), self.next.biases, self.next.preactivations)

        # Apply dropout, zero out preactivations for "dead" nodes, or is it the activations that should be dead?  For sigmoid this makes a difference.
        if self.next.next != None:
            np.multiply(self.next.dropout, self.next.preactivations, self.next.preactivations)

        self.activation_function.apply(self.next.preactivations, self.next.activations)

    def calculate_errors(self, expected_outputs, cost_derivative):
        if self.next == None:
            # Output layer just uses the derivative of the cost function
            self.errors = cost_derivative(self.activations, expected_outputs, self.preactivations)
        else:
            # The hidden layers feed errors backwards
            np.multiply(np.dot(self.next.weights.T, self.next.errors, self.errors), self.activation_function.apply_derivative(self.preactivations), self.errors)

            # Apply dropout, zero out error for "dead" nodes
            np.multiply(self.dropout, self.errors, self.errors)

    def calculate_bias_gradient(self):
        np.sum(self.errors, axis=1, keepdims=True, out=self.bias_gradient)
        return self.bias_gradient
    
    def calculate_weight_gradient(self):
        np.dot(self.errors, self.prev.activations.T, self.weight_gradient)
        return self.weight_gradient

    def update_dropout(self):
        self.dropout = (np.random.rand(self.size, 1) < self.dropout_keep_prob) * 1.

