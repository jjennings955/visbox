import functools
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import pyqtgraph as pg
import cv2

from client import FeatureClient
from util import good_shape, build_imagegrid

class VisualizationWindow(QtGui.QMainWindow):
    def __init__(self):
        super(VisualizationWindow, self).__init__()
        pg.setConfigOptions(imageAxisOrder='row-major')

        frame = pg.QtGui.QFrame()
        layout = pg.QtGui.QGridLayout()
        frame.setLayout(layout)

        self.camera_window = pg.GraphicsLayoutWidget(self)
        self.feature_window = pg.GraphicsLayoutWidget(self)
        self.detailed_feature_window = pg.GraphicsLayoutWidget()

        self.layer_frame = pg.QtGui.QFrame()
        self.layer_layout = pg.QtGui.QHBoxLayout()
        self.layer_frame.setLayout(self.layer_layout)

        layout.addWidget(self.camera_window, 0, 0, 1, 2)
        layout.addWidget(self.build_config_frame(), 1, 0, 1, 1)
        layout.addWidget(self.feature_window, 2, 0, 2, 2)
        layout.addWidget(self.layer_frame, 3, 0, 1, 2)

        layout.setRowStretch(0, 3)
        layout.setRowStretch(1, .1)
        layout.setRowStretch(2, 4)
        layout.setRowStretch(3, 0.1)

        self.setCentralWidget(frame)

        self.setGeometry(0, 200, 1600, 900)

        self.selected_filter = 1

        self.current_layer_dimensions = (512, 7, 7)
        self.ready = True

        self.rows, self.cols = good_shape(self.current_layer_dimensions[0])
        self.feature_client = FeatureClient()

        self.feature_client.prediction_received(self.prediction_received)
        self.feature_client.layerinfo_received(self.layerinfo_received)
        self.feature_client.summary_received(self.summary_received)

        self.feature_client.get_summary()
        self.feature_client.get_layer_info()

        self.last_feature_image = None
        self.video_capture = cv2.VideoCapture(0)
        self.last_frame = None
        self.last_button = None
        self.rois = []
        self.selector = None

        self.build_views()
        self.start_timers()

    def build_config_frame(self):
        self.config_frame = pg.QtGui.QFrame()
        self.config_layout = pg.QtGui.QHBoxLayout()
        self.config_frame.setLayout(self.config_layout)
        self.server_button = pg.QtGui.QPushButton("Start Server")
        self.webcam_button = pg.QtGui.QPushButton("Camera")
        self.video_button = pg.QtGui.QPushButton("Video")
        self.ROI_button = pg.QtGui.QPushButton("ROI")

        self.config_layout.addWidget(self.server_button)
        self.config_layout.addWidget(self.webcam_button)
        self.config_layout.addWidget(self.video_button)
        self.config_layout.addWidget(self.ROI_button)

        self.server_button.clicked.connect(self.server_clicked)
        self.webcam_button.clicked.connect(self.webcam_clicked)
        self.video_button.clicked.connect(self.video_clicked)
        self.ROI_button.clicked.connect(self.ROI_clicked)

        return self.config_frame

    def server_clicked(self, *args, **kwargs):
        from server import run_vgg16
        from multiprocessing import Process
        p = Process(target=run_vgg16)
        p.start()

    def webcam_clicked(self, *args, **kwargs):
        self.video_capture = cv2.VideoCapture(0)

    def video_clicked(self, *args, **kwargs):
        z = QtGui.QFileDialog.getOpenFileName()
        try:
            z = cv2.VideoCapture(z)
            assert z.isOpened()
            self.video_capture = z
        except Exception as e:
            print(e)

    def ROI_clicked(self, *args, **kwargs):
        if not self.selector:
            self.selector = pg.RectROI((5, 5), size=(640, 480))
            self.camera_view.addItem(self.selector)
        else:
            self.camera_view.removeItem(self.selector)
            self.selector = None



    def set_selected_layer(self, layer=None, button=None):
        if button:
            if self.last_button and self.last_button != button:
                self.last_button.setStyleSheet('')
            button.setStyleSheet('QPushButton { background-color: #ff0000; }')
            self.last_button = button

        self.feature_client.change_layer(layer)
        self.feature_client.get_layer_info()

    def summary_received(self, summary):
        self.layer_info = summary['result']

        for i, layer in enumerate(self.layer_info):
            button = pg.QtGui.QPushButton("{}".format(layer))

            button.clicked.connect(functools.partial(self.set_selected_layer, layer=i, button=button))
            self.layer_layout.addWidget(button)

    def layerinfo_received(self, layerinfo):
        self.current_layer_dimensions = layerinfo['shape'][1:]
        self.current_layer_name = layerinfo['name']
        self.rows, self.cols = good_shape(self.current_layer_dimensions[-1])

    def camera_callback(self):
        has_frame, frame = self.video_capture.read()
        first = False
        if has_frame:
            if not np.any(self.last_frame):
                first = True
            self.last_frame = frame[:, :, ::-1]
            self.camera_image.setImage(frame[:, :, ::-1])
        if first:
            self.do_prediction()

    def do_prediction(self):
        if np.any(self.last_frame) and self.ready:
            if self.selector:
                im = self.selector.getArrayRegion(self.last_frame, self.camera_image)
            else:
                im = self.last_frame
            self.lastframe_image.setImage(im)
            self.feature_client.predict(im)

    def start_timers(self):
        self.camera_timer = QtCore.QTimer()
        self.camera_timer.timeout.connect(self.camera_callback)
        self.camera_timer.start(10)

        self.feature_server_timer = QtCore.QTimer()
        self.feature_server_timer.timeout.connect(self.feature_client.check)
        self.feature_server_timer.start(50)

    def build_views(self):
        self.camera_view = self.camera_window.addViewBox()
        self.lastframe_view = self.camera_window.addViewBox()
        self.features_view = self.feature_window.addViewBox()

        self.detailed_feature_view = self.camera_window.addViewBox()
        self.features_view.setRange(QtCore.QRectF(0, 0, 1600, 1600))
        self.camera_view.setAspectLocked(True)
        self.features_view.setAspectLocked(True)
        self.detailed_feature_view.setAspectLocked(True)
        self.lastframe_view.setAspectLocked(True)

        self.camera_view.invertY()
        self.camera_view.setMouseEnabled(False, False)
        self.detailed_feature_view.invertY()
        self.features_view.invertY()
        self.lastframe_view.invertY()
        self.lastframe_view.setMouseEnabled(False, False)



        self.camera_image = pg.ImageItem(border='w')
        self.lastframe_image = pg.ImageItem(border='w')
        self.features_image = pg.ImageItem(border='w')
        self.detailed_features_image = pg.ImageItem(border='w')



        self.features_image.mouseClickEvent = self.select_filter

        self.features_image.setLevels([0, 255])
        self.detailed_features_image.setLevels([0, 255])

        self.camera_view.addItem(self.camera_image)
        self.features_view.addItem(self.features_image)
        self.detailed_feature_view.addItem(self.detailed_features_image)
        self.lastframe_view.addItem(self.lastframe_image)

        self.features_view.enableAutoRange()
        #self.layers_view.enableAutoRange()
        self.detailed_feature_view.enableAutoRange()
        self.lastframe_view.enableAutoRange()

    def select_filter(self, event):
        x = event.pos().x()
        y = event.pos().y()
        row = y // self.current_layer_dimensions[0]
        col = x // self.current_layer_dimensions[1]

        n = self.cols*row + col
        if n < self.current_layer_dimensions[2]:
            print("Selected filter {}".format(n))
            self.selected_filter = int(n)

    def prediction_received(self, result, *args, **kwargs):
        result = result['result'].squeeze()
        transposed = np.transpose(result, (2, 0, 1))
        image = build_imagegrid(image_list=transposed, n_rows=self.rows, n_cols=self.cols)
        image /= np.max(image)
        self.last_feature_image = np.uint8(255.0 * image)
        self.features_image.setImage(self.last_feature_image)
        self.features_view.autoRange()
        self.lastframe_view.autoRange()
        if self.selected_filter < transposed.shape[0]:
            self.detailed_features_image.setImage(transposed[self.selected_filter])
            self.detailed_feature_view.autoRange()
        self.do_prediction()


if __name__ == '__main__':
    import sys

    app = QtGui.QApplication([])
    z = VisualizationWindow()
    z.show()
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

