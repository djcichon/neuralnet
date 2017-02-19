import numpy as np

#TODO: Back propagation by batch
class Network:
    """ A fully connected Neural Network """

    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.biases = self._initialize_biases()
        self.weights = self._initialize_weights()
        self.activations = self._initialize_activations()
        self.preactivations = self._initialize_activations()
        self.errors = self._initialize_errors()

    def _initialize_biases(self):
        """ Create a list of numpy arrays for the biases; one per neuron
            Note: No list is created for the input layer """

        biases = []

        for size in self.layer_sizes[1:]:
            biases.append(np.random.normal(size=(size, 1)))

        return biases

    def _initialize_weights(self):
        """ Create a list of numpy arrays for the weights (layer_size x prev_layer_size)
            Note: No list is created for the input layer """

        weights = []

        for index in range(1, len(self.layer_sizes)):
            size = self.layer_sizes[index]
            prev_size = self.layer_sizes[index-1]

            weights.append(np.random.normal(size=(size, prev_size)))

        return weights

    def _initialize_activations(self):
        """ Create a list of numpy arrays for activations; one per neuron """
        activations = []

        for size in self.layer_sizes:
            activations.append(np.zeros((size, 1)))

        return activations

    def _initialize_errors(self):
        """ Create a list of numpy arrays for errors; one per neuron
            Note: No list is created for the input layer """

        errors = []

        for size in self.layer_sizes[1:]:
            errors.append(np.zeros((size, 1)))

        return errors



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
        self.preactivations[layer_index] = preactivations

        return self.sigmoid(preactivations)



    def back_propagation(self, expected_outputs, learning_rate = 1):
        self._calculate_errors(expected_outputs)

        # Loop from output layer back to first hidden layer
        for layer_index in range(len(self.layer_sizes) - 1, 0, -1):
            self._update_weights(layer_index, learning_rate)
            self._update_biases(layer_index, learning_rate)

    def _calculate_errors(self, expected_outputs):
        self._calculate_output_layer_errors(expected_outputs)

        # Loop from last hidden layer back to the first hidden layer
        for layer_index in range(len(self.layer_sizes) - 2, 0, -1):
            self._calculate_hidden_layer_errors(layer_index)

    def _calculate_output_layer_errors(self, expected_outputs):
        # errors = (actual - expected) * slope of activation function
        self.errors[-1] = (self.activations[-1] - expected_outputs) * self.sigmoid_prime(self.preactivations[-1])

    def _calculate_hidden_layer_errors(self, layer_index):
        # errors = next_weights_transposed * next_errors * slope of activation function
        dot = np.dot(np.transpose(self.weights[layer_index]), self.errors[layer_index])
        self.errors[layer_index-1] =  dot * self.sigmoid(self.preactivations[layer_index])

    def _update_weights(self, layer_index, learning_rate):
        dot = np.dot(self.errors[layer_index-1], np.transpose(self.activations[layer_index-1]))

        #TODO: divide by batch size here
        self.weights[layer_index-1] -= learning_rate * dot

    def _update_biases(self, layer_index, learning_rate):
        #TODO: divide by batch size here
        self.biases[layer_index-1] -= learning_rate * self.errors[layer_index - 1]



    @staticmethod
    def sigmoid(preactivations):
        return 1 / (1 + np.exp(-preactivations))

    @staticmethod
    def sigmoid_prime(preactivations):
        e_to_minus_x = np.exp(-preactivations)

        return e_to_minus_x / np.square(1 + e_to_minus_x)

