import numpy as np
from layer import Layer
from cost import *
from util import *

class Network:
    """ A fully connected Neural Network """

    def __init__(self, layer_sizes, cost_function=CrossEntropy):
        self._initialize_layers(layer_sizes)
        self.cost_function = cost_function

    def _initialize_layers(self, layer_sizes):
        # Create all the layers (unconnected)
        self.layers = [Layer(layer_size) for layer_size in layer_sizes]

        # Connect all the layers to their neighbors
        for i in range(0, len(self.layers)):
            prev_layer = self.layers[i - 1] if i > 0 else None
            next_layer = self.layers[i + 1] if i + 1 < len(self.layers) else None

            self.layers[i].connect_to_neighbors(prev_layer, next_layer)

    def feed_forward(self, inputs):
        self.layers[0].activations = inputs

        for layer in self.layers[:len(self.layers) - 1]:
            layer.forward()

        return self.layers[-1].activations

    def back_propagation(self, batch):
        self.feed_forward(batch[0])
        self._calculate_errors(batch[1])

        for layer in self.layers[1:len(self.layers)]:
            layer.bias_gradient = layer.calculate_bias_gradient()
            layer.weight_gradient = layer.calculate_weight_gradient()

    def SGD(self, training_data, test_data, epochs = 100, learning_rate=1.0, regularization=0.0, batch_size=1):
        """ Trains this network using stochastic gradient descent.
            training_data[0] is expected to have a list of inputs
            training_data[1] is expected to have a list of expected outputs
            test_data is structured similarly, but is used to validate the training """

        for epoch in range(1, epochs+1):
            print("Beginning epoch " + str(epoch))

            for batch in Network._get_shuffled_batches(training_data, batch_size):
                self.back_propagation(batch)

                for layer in self.layers[1:len(self.layers)]:
                    layer.biases -= float(learning_rate) / batch_size * layer.bias_gradient

                    layer.weights -= (float(learning_rate) / batch_size * layer.weight_gradient +
                        (learning_rate * regularization / training_data[0].shape[1]) * layer.weights)


            self._report_performance(test_data)

    @staticmethod
    def _get_shuffled_batches(data, batch_size):
	total_examples = data[0].shape[1]
	permutation = list(np.random.permutation(total_examples))

	shuffled_X = data[0][:, permutation]
	shuffled_Y = data[1][:, permutation]

	return [[shuffled_X[:, index:index + batch_size], shuffled_Y[:, index:index + batch_size]] for index in range(0, total_examples, batch_size)]

    def _report_performance(self, data):
        result = self.feed_forward(data[0])

        # Calculate the expected/actual number which is predicted
        expected = np.argmax(data[1], axis=0)
        actual = np.argmax(result, axis=0)

        # Combine actual/expected into a list where True = correct, False = incorrect
        correct_predictions = (actual == expected)

        # Aggregate this list into a number of correct predictions
        correct_count = np.sum(correct_predictions)

        print("Number correct: " + str(correct_count) + " of " + str(data[0].shape[1]))

    def _reset_gradients(self):
        for layer in self.layers[1:len(self.layers)]:
            layer.reset_gradients()


    def _calculate_errors(self, expected_outputs):
        for layer in reversed(self.layers[1:]):
            layer.calculate_errors(expected_outputs, self.cost_function.cost_derivative)
