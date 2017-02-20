import mnist
import network

training, test = mnist.load()
n = network.Network([28*28, 30, 10])
n.SGD(training, test, 30, 3.0, 10)

