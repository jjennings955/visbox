import numpy as np
import zmq
from keras.engine import Model
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
#from keras.applications import ResNet50 as VGG16
#from keras.applications.resnet50 import preprocess_input
import cv2

class FeatureComputer(object):
    def __init__(self, bind_str="tcp://127.0.0.1:5560", parent_model=None, layer=None):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PAIR)
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


def run_vgg16():
    model = VGG16(False, "imagenet")
    z = FeatureComputer(parent_model=model)
    z.run()

if __name__ == "__main__":
#    from vggface import load_vggface
    model = VGG16(False, "imagenet")
    #model = load_vggface('./models/vggface/vgg-face-keras.h5')
    z = FeatureComputer(parent_model=model)
    z.run()
