import asyncio
import json
import websockets
from observer import Observer
from queue import Queue
from threading import Thread
from autobahn.asyncio.websocket import WebSocketServerProtocol, WebSocketServerFactory

class WebServerProtocol(WebSocketServerProtocol):
    def onOpen(self):
        print("WebSocket connection open.")
        WebServer.websockets.append(self)

    def onClose(self, wasClean, code, reason):
        print("WebSocket connection closed: {0}".format(reason))
        WebServer.websockets.remove(self)

class WebServer(Observer):
    #TODO: I'd like a better method of sharing websockets, but this works.
    #TODO: The other solutions I can think of are to use a singleton or a global.
    websockets = []
    notifications = Queue()

    def start(self):
        # Start the neural network
        t = Thread(target=self.run_neural_network)
        t.start()

        # Start the web server
        print("Starting web server")
        factory = WebSocketServerFactory(u"ws://127.0.0.1:5678")
        factory.protocol = WebServerProtocol

        loop = asyncio.get_event_loop()
        coro = loop.create_server(factory, '0.0.0.0', 5678)
        server = loop.run_until_complete(coro)

        try:
            loop.run_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.close()
            loop.close()

    def run_neural_network(self):
        raise Exception("Override run_neural_network with your own neural net code");

    def notify(self, observable, *args, **kwargs):
        json_message = json.dumps({'type': args[0], 'value': args[1]})
        self.notifications.put(json_message.encode('utf-8'))
        
        if len(WebServer.websockets) > 0:
            while not self.notifications.empty():
                notification = self.notifications.get()

                for connection in WebServer.websockets:
                    connection.sendMessage(notification, False)
