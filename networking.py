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

import yaml
import numpy as np
import zmq
import zmq.auth
from zmq.auth.thread import ThreadAuthenticator

from keras.engine import Model
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
import cv2

class FeatureComputer(object):
    def __init__(self, bind_str="tcp://127.0.0.1:5560", parent_model=None, layer=None, logins=None):
        self.context = zmq.Context.instance()
        self.auth = ThreadAuthenticator(self.context)
        self.auth.start()
        #auth.allow('127.0.0.1')
        self.auth.configure_plain(domain='*', passwords=logins)
        self.socket = self.context.socket(zmq.PAIR)
        self.socket.plain_server = True
        self.socket.bind(bind_str)
        self.parent_model = parent_model
        self.curr_model = parent_model

        if not layer:
            self.layer = 0 #len(self.parent_model.layers) - 1
        else:
            self.layer = layer

    def change_layer(self, *args, **kwargs):
        print("Changing layer")
        print(args, kwargs)
        self.layer = kwargs.get('layer', self.layer)
        self.curr_model = Model(input=[self.parent_model.layers[0].input],
                           output=[self.parent_model.layers[self.layer].output])
        print(self.layer)
        self.socket.send_pyobj({'type': 'layer_changed', 'success': True}, zmq.NOBLOCK)

    def get_summary(self, *args, **kwargs):
        self.socket.send_pyobj({'type': 'summary', 'result' : None}, zmq.NOBLOCK)

    def do_predict(self, *args, **kwargs):
        # TODO: Make this configurable
        input = kwargs.pop('input', np.zeros((1,224,224,3)))

        resized = np.float64(cv2.resize(input, (224, 224)))
        preprocessed = preprocess_input(np.expand_dims(resized, axis=0), dim_ordering='tf')

        result = self.curr_model.predict(preprocessed, verbose=0)
        self.socket.send_pyobj({'type': 'prediction', 'result': result}, zmq.NOBLOCK)

    def do_layerinfo(self, *args, **kwargs):
        self.socket.send_pyobj({'type': 'layer_info', 'shape': self.curr_model.get_output_shape_for((1, 224, 224, 3)), 'name' : self.parent_model.layers[self.layer].name }, zmq.NOBLOCK)

    def do_summary(self, *args, **kwargs):
        self.socket.send_pyobj({'type': 'summary', 'result': [layer.name for layer in self.parent_model.layers]}, zmq.NOBLOCK)

    def run(self):
        self.running = True
        while self.running:
            message = self.socket.recv_pyobj()
            try:
                if message['type'] == "change_layer":
                    self.change_layer(**message)
                if message['type'] == 'predict':
                    self.do_predict(**message)
                if message['type'] == 'layer_info':
                    self.do_layerinfo(**message)
                if message['type'] == 'summary':
                    self.do_summary(**message)
            except:
                print("Error sending, ignoring")

def default_config():
    settings = {}
    settings['bind_addr'] = 'tcp://127.0.0.1:5560'
    settings['logins'] = {'admin' : 'secret'}
    with open('server.yaml', 'r') as fd:
        print("### WARNING - RUNNING WITH DEFAULT PASSWORD ###")
        return yaml.dump(settings, fd)


def load_config(fname):
    try:
        with open(fname, 'r') as fd:
            return yaml.load(fd)
    except:
        return default_config()
