import mnist
from network import Network
from layer import Layer
from activation import *

training, test = mnist.load()

n = Network()
n.add_layer(Layer(28*28))
n.add_layer(Layer(30, Sigmoid))
n.add_layer(Layer(30, Sigmoid))
n.add_layer(Layer(10, Sigmoid))

n.SGD(training, test, 100, 0.1, 0, 1024)

