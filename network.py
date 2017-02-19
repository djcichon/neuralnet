import numpy as np

class Network:
    """ A fully connected Neural Network """

    def __init__(self, layer_sizes, activation_function = ActivationFunctions.sigmoid):
        self.layer_sizes = layer_sizes
        self.biases = self._initialize_biases()
        self.weights = self._initialize_weights()
        self.activations = self._initialize_activations()
        self.activation_function = activation_function

    def _initialize_biases(self):
        """ Create a list of numpy arrays for the biases
            Note: No list is created for the input layer """

        biases = []

        for size in self.layer_sizes[1:]:
            biases.append(np.random.normal(size=(size, 1)))

        return biases

    def _initialize_weights(self):
        """ Create a list of numpy arrays for the weights
            Note: No list is created for the input layer """

        weights = []

        for index in range(1, len(self.layer_sizes)):
            size = self.layer_sizes[index]
            prev_size = self.layer_sizes[index-1]

            weights.append(np.random.normal(size=(size, prev_size)))

        return weights

    def _initialize_activations(self):
        activations = []

        for size in self.layer_sizes:
            activations.append(np.zeros((size, 1)))

        return activations

    def feed_forward(self, inputs):
        self._set_inputs(inputs)

        for layer_index in range(1, len(self.activations)):
            self.activations[layer_index] = self._calculate_activations(layer_index)

        return self.activations[-1]

    def _set_inputs(self, inputs):
        if(len(inputs) != self.layer_sizes[0]):
            raise Exception('Expected inputs of length ' + str(self.layer_sizes[0]) + ' but was ' + str(len(inputs)))

        # Copy from inputs into activations
        self.activations[0][:] = inputs

    def _calculate_activations(self, layer_index):
        preactivations = np.dot(self.weights[layer_index-1], self.activations[layer_index-1]) + self.biases[layer_index-1]

        return self.activation_function(preactivations)

class ActivationFunctions:

    @staticmethod
    def sigmoid(preactivations):
        return 1 / (1 + np.exp(-preactivations))

