from keras.engine import Model
from keras.applications import VGG16
import h5py
import re
import numpy as np

patt = re.compile(r'conv([0-9]+)_([0-9]+)')

def convert_layername(name):
    return patt.sub(r"block\1_conv\2", name)

def load_vggface(weights_hd5):
    base_model = VGG16(weights=None, include_top=False)
    weights = h5py.File(weights_hd5, 'r')
    print(weights['/'].keys())
    for i in weights['/'].keys():
        if i.startswith('conv'):
            print("{face}->{keras} (weights={weights}, biases={biases})".format(face=i, keras=convert_layername(i), weights=('/' + i + '/' + i + '_W'), biases=('/' + i + '/' + i + '_b')))
            l = []
            w = np.transpose(weights['/' + i + '/' + i + '_W'], (3, 2, 1, 0))
            b = weights['/' + i + '/' + i + '_b']
            l.append(w)
            l.append(b)
            base_model.get_layer(convert_layername(i)).set_weights(l)
    return base_model

#x = load_vggface('./vgg-face-keras.h5')
#print(x.predict(np.random.randn(1,224,224,3)))