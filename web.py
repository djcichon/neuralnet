import mnist
from network import Network
from webserver import WebServer

class MyWebServer(WebServer):

    def run_neural_network(self):
        training, test = mnist.load()

        network = Network([28*28, 100, 10])
        network.register_observer(self)

        network.SGD(training, test, 100, 3.0, 10)

ws = MyWebServer()
ws.start()
