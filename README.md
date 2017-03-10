# visbox (tentative name, apparently it's already a thing)
This is a tool I created to visualize neural networks in keras.
Heavily inspired by (and functionally inferior to) https://github.com/yosinski/deep-visualization-toolbox

## Features
- Runs at approximately 20-30+ fps on various GPUs (My laptop's Quadro M1000M, a Geforce 760, a K40 server (over LAN), 970M on a Macbook Pro)
- Runs at about 0.5-2 FPS on CPU (on an i5 4670k)
- Only 500 lines of code, so you can probably figure out what's going on (it's mostly GUI stuff)
- Server can be run remotely


# Code Breakdown
- main.py - Gui stuff
- server.py - Runs the neural network part + backend communication
- client.py - Communicates with the server

## Windows
```bash
conda create --name visbox python=3.5
activate visbox
pip install git+https://github.com/fchollet/keras.git
conda install pyzmq==16.0.2
conda install numpy scipy h5py pyyaml
conda install -c menpo opencv3
pip install pyqt5
pip install tensorflow # tensorflow-gpu if you have a gpu. You can also use theano.
conda install -c anaconda pyqt=4.11.4
```

## Linux/OS X (untested)
```bash
conda create --name visbox python=3.5
source activate visbox
pip install git+https://github.com/fchollet/keras.git
conda install pyzmq==16.0.2
conda install numpy scipy h5py pyyaml
conda install -c menpo opencv3
pip install pyqt5
pip install tensorflow # tensorflow-gpu if you have a gpu. You can also use theano.
conda install -c anaconda pyqt=4.11.4
```

## Usage
```
activate visbox
python server.py # The first time you run this it will download the weights for VGG16
python main.py # in another terminal/screen/whatever
```

In the GUI:
- Click "Connect" (you should see the layer names appear at the bottom)
- Click "Webcam" or "Video" to select a video source
- Select a layer you find interesting at the bottom
- Click on a feature in the grid to get a better view

Optional:
- Click ROI to enable a region of interest selector from the video stream
- There is a scroll bar for fastforwarding through a video, that does nothing when you're using a webcam (and probably shouldn't be visible/enabled)
- You can run the server from the GUI, but it's not recommended (especially the first time)


## Known issues
- Server is extremely insecure (no authentication + pickle).
- There are some sketchy "catch Exception" blocks in the code that need to be made more specific
- The features grids need actual grid lines
- Doesn't work with fancy architectures (anything with branching i.e. Resnet)
- I suck at interface design
- ROI selector makes things slow
- None of the cool visualization algorithms are implemented
- Video capture stuff probably shouldn't be happening in the GUI event loop