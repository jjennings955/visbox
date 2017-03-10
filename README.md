# visbox (tentative name, apparently it's already a thing)
This is a tool I created to visualize neural networks in keras.
Heavily inspired (and functionally inferior to) https://github.com/yosinski/deep-visualization-toolbox

## Features
Runs at approximately 20-30 fps on various GPUs (My laptop's Quadro M1000M, a Geforce 760, a K40 server (over LAN), 970M on a Macbook Pro)
Runs at about 0.5-1 FPS on CPU (on an i5 4670k)
Server can be run remotely

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
python main.py
```

Click "Run server" (may take a while depending on GPU/etc). Optionally, you can run server.py in another terminal (or on another server).
Click "Connect"
Click "Webcam" or "Video" to select a video source
Select a layer you find interesting at the bottom
Click on a feature in the grid to get a better view

Optional:
Click ROI to enable a region of interest selector from the video stream
There is a scroll bar for fastforwarding through a video, that does nothing when you're using a webcam (and probably shouldn't be visible/enabled)


## Known issues
- Server is not secured
- The features grids need grid lines
- Doesn't work with fancy architectures (anything with branching i.e. Resnet)
- I suck at interface design
- ROI selector makes things slow
- None of the cool visualization algorithms are implemented 
