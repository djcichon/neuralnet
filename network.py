import numpy as np
import math
import observable

class Network(observable.Observable):
    """ A fully connected Neural Network """
    #TODO: Data reporting/visualization
    #TODO: Regularization
    #TODO: Cost functions
    #TODO: Activation functions
    #TODO: Layer types (softmax first, convolutional later)

    def __init__(self, layer_sizes):
        observable.Observable.__init__(self)

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
            biases.append(np.random.randn(size, 1) / math.sqrt(size))

        return biases

    def _initialize_weights(self):
        """ Create a list of numpy arrays for the weights (layer_size x prev_layer_size)
            Note: No list is created for the input layer """

        weights = []

        for index in range(1, len(self.layer_sizes)):
            size = self.layer_sizes[index]
            prev_size = self.layer_sizes[index-1]

            weights.append(np.random.randn(size, prev_size)/math.sqrt(size * prev_size))

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


    def SGD(self, training_data, test_data, epochs = 100, learning_rate=1.0, batch_size=1):
        """ Trains this network using stochastic gradient descent.
            training_data[0] is expected to have a list of inputs
            training_data[1] is expected to have a list of expected outputs
            test_data is structured similarly, but is used to validate the training """

        for epoch in range(1, epochs+1):
            print("Beginning epoch " + str(epoch))
            self.notify_observers("Beginning epoch " + str(epoch))

            for batch in Network._get_shuffled_batches(training_data, batch_size):
                bias_gradient, weight_gradient = self.back_propagation(batch)

                for layer_index in range(1, len(self.layer_sizes)):
                    self.biases[layer_index-1] -= float(learning_rate) / batch_size * bias_gradient[layer_index-1]
                    self.weights[layer_index-1] -= float(learning_rate) / batch_size * weight_gradient[layer_index-1]

            self._report_performance(test_data)

    @staticmethod
    def _get_shuffled_batches(data, batch_size):
        zipped_data = list(zip(data[0], data[1]))
        np.random.shuffle(zipped_data)

        return [zipped_data[index:index+batch_size] for index in range(0, len(zipped_data), batch_size)]

    def _report_performance(self, data):
        correct_count = 0
        for image, label in zip(data[0], data[1]):
            result = self.feed_forward(image)

            if np.argmax(result) == np.argmax(label):
                correct_count += 1

        print("Number correct: " + str(correct_count) + " of " + str(len(data[1])))
        self.notify_observers("Number correct: " + str(correct_count) + " of " + str(len(data[1])))


        
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



    def back_propagation(self, batch):
        bias_gradient = [np.zeros(biases.shape) for biases in self.biases]
        weight_gradient = [np.zeros(weights.shape) for weights in self.weights]

        for image, label in batch:
            self.feed_forward(image)
            self._calculate_errors(label)

            for layer_index in range(1, len(self.layer_sizes)):
                bias_gradient[layer_index-1] += self._calculate_bias_gradient(layer_index)
                weight_gradient[layer_index-1] += self._calculate_weight_gradient(layer_index)

        return bias_gradient, weight_gradient

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
        dot = np.dot(self.weights[layer_index].T, self.errors[layer_index])
        self.errors[layer_index-1] = dot * self.sigmoid_prime(self.preactivations[layer_index])

    def _calculate_bias_gradient(self, layer_index):
        return self.errors[layer_index-1]
    
    def _calculate_weight_gradient(self, layer_index):
        return np.dot(self.errors[layer_index-1], self.activations[layer_index-1].T)


    @staticmethod
    def sigmoid(preactivations):
        return 1 / (1 + np.exp(-preactivations))

    @staticmethod
    def sigmoid_prime(preactivations):
        sig = Network.sigmoid(preactivations)
        return sig * (1 - sig)
