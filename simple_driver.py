from network import Network
from layer import Layer
from activation import *

training = [np.array([[0, 1, 1, 0, 0],
                      [1, 1, 0, 1, 1],
                      [0, 1, 0, 1, 0]]),
            np.array([[0, 1, 0, 1, 0],
                      [1, 1, 1, 0, 0],
                      [1, 1, 1, 0, 0]])]
test = [np.array([[0, 1, 1, 0, 0],
                  [1, 1, 0, 1, 1],
                  [0, 1, 0, 1, 0]]),
        np.array([[0, 1, 0, 1, 0],
                  [1, 1, 1, 0, 0],
                  [1, 1, 1, 0, 0]])]

n = Network()
n.add_layer(Layer(3))
n.add_layer(Layer(5, Tanh))
n.add_layer(Layer(3, Sigmoid))

n.SGD(training, test, 1, 0.1, 0, 5, gradient_check=True)

