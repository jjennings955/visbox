import numpy as np
import zmq


def nocallback(*args, **kwarsg):
    pass

class FeatureClient(object):
    def __init__(self, connect_str="tcp://127.0.0.1:5560", username=None, password=None):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PAIR)
        self.socket.plain_username = bytes(username, 'ascii')
        self.socket.plain_password = bytes(password, 'ascii')
        self.socket.connect(connect_str)
        self.connect(connect_str)
        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)
        self.running = False
        self.prediction_callback = nocallback
        self.summary_callback = nocallback
        self.layerinfo_callback = nocallback

    def connect(self, connect_str):
        self.socket.connect(connect_str)

    def predict(self, input):
        self.socket.send_pyobj({'type' : 'predict', 'input' : input})

    def get_summary(self):
        self.socket.send_pyobj({'type': 'summary' })

    def get_layer_info(self):
        self.socket.send_pyobj({'type': 'layer_info' })

    def change_layer(self, n):
        self.socket.send_pyobj({'type': 'change_layer', 'layer' : n })

    def prediction_received(self, callback):
        self.prediction_callback = callback

    def summary_received(self, callback):
        self.summary_callback = callback

    def layerinfo_received(self, callback):
        self.layerinfo_callback = callback

    def check(self):
        socks = dict(self.poller.poll(1))
        if socks:
            if socks.get(self.socket) == zmq.POLLIN:
                z = self.socket.recv_pyobj(zmq.NOBLOCK)
                if z['type'] == 'prediction':
                    self.prediction_callback(z)
                if z['type'] == 'summary':
                    self.summary_callback(z)
                if z['type'] == 'layer_info':
                    self.layerinfo_callback(z)

if __name__ == "__main__":
    x = FeatureClient()