import numpy as np
from layer import Layer
from cost import *
from activation import *

class Network:
    """ A fully connected Neural Network """

    def __init__(self, cost_function=CrossEntropy):
        self.layers = []
        self.cost_function = cost_function

    def add_layer(self, layer):
        prev_layer = self.layers[-1] if len(self.layers) > 0 else None
        layer.connect_to_previous_layer(prev_layer)

        self.layers.append(layer)

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

    def SGD(self, training_data, test_data, epochs = 100, learning_rate=1.0, regularization=0.0, batch_size=1, gradient_check=False):
        """ Trains this network using stochastic gradient descent.
            training_data[0] is expected to have a list of inputs
            training_data[1] is expected to have a list of expected outputs
            test_data is structured similarly, but is used to validate the training """

        for epoch in range(1, epochs+1):
            print("Beginning epoch " + str(epoch))


            for batch in Network._get_shuffled_batches(training_data, batch_size):
                for layer in self.layers:
                    layer.initialize_matrices(batch[0].shape[1])

                self.update_dropout()
                self.back_propagation(batch)

                if gradient_check is True:
                    self.gradient_check(batch)

                for layer in self.layers[1:len(self.layers)]:
                    # Update the bias (this is made obtuse for performance reasons)
                    np.multiply(float(learning_rate), layer.bias_gradient, out=layer.bias_gradient)
                    np.subtract(layer.biases, layer.bias_gradient, out=layer.biases)

                    # Update the weights (this is made obtuse for performance reasons)
                    np.multiply(float(learning_rate), layer.weight_gradient, out=layer.weight_gradient)
                    np.subtract(layer.weights, (learning_rate * regularization / training_data[0].shape[1]) * layer.weights, layer.weights)
                    np.subtract(layer.weights, layer.weight_gradient, out=layer.weights)

            self._report_performance(training_data, "TRAIN")
            self._report_performance(test_data, "TEST ")

    def calculate_cost(self, batch):
        predictions = self.feed_forward(batch[0])

        return self.cost_function.cost(predictions, batch[1])

    def gradient_check(self, batch):
        epsilon = 1e-7

        gradient_approxes = []
        gradients = []

        for layer in self.layers[1:]:
            weights = layer.weights
            biases = layer.biases

            total_gradient_error = 0
            # Loop over each weight
            for i in range(0, weights.shape[0]):
                for j in range(0, layer.weights.shape[1]):
                    orig_weight = weights[i][j]

                    weights[i][j] = orig_weight - epsilon
                    cost_before = self.calculate_cost(batch)
                    
                    weights[i][j] = orig_weight + epsilon
                    cost_after = self.calculate_cost(batch)
                    
                    weights[i][j] = orig_weight

                    gradient = layer.weight_gradient[i][j]
                    gradient_approx = (cost_after - cost_before) / (2 * epsilon)

                    gradients.append(gradient)
                    gradient_approxes.append(gradient_approx)

            # Loop over each bias
            for i in range(0, biases.shape[0]):
                orig_bias = biases[i][0]

                biases[i][0] = orig_bias - epsilon
                cost_before = self.calculate_cost(batch)
                
                biases[i][0] = orig_bias + epsilon
                cost_after = self.calculate_cost(batch)
                
                biases[i][0] = orig_bias

                gradient = layer.bias_gradient[i][0]
                gradient_approx = (cost_after - cost_before) / (2 * epsilon)

                gradients.append(gradient)
                gradient_approxes.append(gradient_approx)

        np_gradient_approxes = np.array(gradient_approxes)
        np_gradients = np.array(gradients)

        numerator = np.linalg.norm(np_gradients - np_gradient_approxes)
        denominator = np.linalg.norm(np_gradients) + np.linalg.norm(np_gradient_approxes)
        error = numerator / denominator

        print("Total error: " + str(error))


    def update_dropout(self):
        for layer in self.layers[1:len(self.layers) - 1]:
            layer.update_dropout()

    @staticmethod
    def _get_shuffled_batches(data, batch_size):
        X = data[0]
        Y = data[1]
	total_examples = X.shape[1]
	#permutation = list(np.random.permutation(total_examples))

        rng_state = np.random.get_state()
        np.random.shuffle(X.T)
        np.random.set_state(rng_state)
        np.random.shuffle(Y.T)
	#shuffled_X = data[0][:, permutation]
	#shuffled_Y = data[1][:, permutation]

	return [[X[:, index:index + batch_size], Y[:, index:index + batch_size]] for index in range(0, total_examples, batch_size)]

    def _report_performance(self, data, label):
        total_examples = data[0].shape[1]

        for layer in self.layers:
            layer.initialize_matrices(total_examples)

        result = self.feed_forward(data[0])

        # Calculate the expected/actual number which is predicted
        expected = np.argmax(data[1], axis=0)
        actual = np.argmax(result, axis=0)

        # Combine actual/expected into a list where True = correct, False = incorrect
        correct_predictions = (actual == expected)

        # Aggregate this list into a number of correct predictions
        correct_count = np.sum(correct_predictions)

        print("Accuracy on " + label + ": " + "{0:.2f}".format(100. * correct_count / total_examples) + "%")

    def _reset_gradients(self):
        for layer in self.layers[1:len(self.layers)]:
            layer.reset_gradients()


    def _calculate_errors(self, expected_outputs):
        for layer in reversed(self.layers[1:]):
            layer.calculate_errors(expected_outputs, self.cost_function.cost_derivative)

