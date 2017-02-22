import asyncio
import websockets
from observer import Observer
from queue import Queue
from threading import Thread

class WebServer(Observer):
    messages = Queue()

    def start(self):
        # Start the neural network
        t = Thread(target=self.run_neural_network)
        t.start()

        # Start the webserver
        start_server = websockets.serve(WebServer.send_queued_messages, '127.0.0.1', 5678)

        # Keep server alive, even when network finishes.  
        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()

    def run_neural_network(self):
        raise Exception("Override run_neural_network with your own neural net code");

    def notify(self, observable, *args, **kwargs):
        self.messages.put(str(*args))

    @staticmethod
    @asyncio.coroutine
    def send_queued_messages(websocket, path):
        print("A user connected.")

        messages = WebServer.messages

        while True:
            message = messages.get()
            yield from websocket.send(message)