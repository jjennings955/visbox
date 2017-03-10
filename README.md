# Windows
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

# Linux/OS X (untested)
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

# Usage
```
activate visbox
python main.py
```

Click "Run server" (may take a while depending on GPU/etc). Optionally, you can run server.py in another terminal (or on another server).
Click "Connect"
Click "Webcam" or "Video" to select a video source

You can select a layer at the bottom.
Click a feature to enlarge it.

# Known issues
- Server is not secured
- The features grids need grid lines
- Doesn't work with fancy architectures (anything with branching i.e. Resnet)
- I suck at interface design
- ROI selector makes things slow
