import mnist
import network

training, test = mnist.load()
n = network.Network([28*28, 50, 25, 10])
n.SGD(training, test, 100, 0.1, 5, 10)
