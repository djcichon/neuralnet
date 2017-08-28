import mnist
import network
from activation import *

training, test = mnist.load()

n = network.Network()
n.add_layer(28*28)
n.add_layer(100, ReLU)
n.add_layer(10, Sigmoid)

n.SGD(training, test, 100, 0.1, 5, 10)

