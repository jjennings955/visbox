import functools
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import pyqtgraph as pg
import cv2

from client import FeatureClient
from util import good_shape, build_imagegrid, load_config

class VisualizationWindow(QtGui.QMainWindow):
    def __init__(self):
        super(VisualizationWindow, self).__init__()
        pg.setConfigOptions(imageAxisOrder='row-major')
        self.settings = load_config()

        frame = pg.QtGui.QFrame()
        layout = pg.QtGui.QGridLayout()
        frame.setLayout(layout)

        self.camera_window = pg.GraphicsLayoutWidget(self)
        self.feature_window = pg.GraphicsLayoutWidget(self)
        self.detailed_feature_window = pg.GraphicsLayoutWidget()

        self.layer_frame = pg.QtGui.QFrame()
        self.layer_layout = pg.QtGui.QGridLayout()
        self.layer_frame.setLayout(self.layer_layout)

        layout.addWidget(self.camera_window, 0, 0, 1, 2)
        layout.addWidget(self.build_config_frame(), 1, 0, 1, 1)
        layout.addWidget(self.feature_window, 2, 0, 2, 2)
        layout.addWidget(self.layer_frame, 3, 0, 2, 2)

        layout.setRowStretch(0, 30)
        layout.setRowStretch(1, 1)
        layout.setRowStretch(2, 40)
        layout.setRowStretch(3, 1)

        self.setCentralWidget(frame)

        self.setGeometry(0, 200, 1600, 900)

        self.selected_filter = 1

        self.current_layer_dimensions = (512, 7, 7)
        self.ready = True

        self.rows, self.cols = good_shape(self.current_layer_dimensions[0])


        self.last_feature_image = None
        self.video_capture = None
        self.last_frame = None
        self.last_button = None
        self.feature_server_timer = None
        self.rois = []
        self.selector = None

        self.build_views()
        self.start_timers()
        self.process_settings()


    def process_settings(self):
        if self.settings.get('auto_connect', False):
           self.init_client(self.settings['servers'][self.settings.get('default_server', 'local')])
           self.start_feature_server_timer()
        if self.settings.get('auto_start_camera', False):
            self.webcam_clicked()


    def init_client(self, connect_str=None):
        print("init_client", connect_str)
        self.feature_client = FeatureClient(connect_str)
        self.feature_client.prediction_received(self.prediction_received)
        self.feature_client.layerinfo_received(self.layerinfo_received)
        self.feature_client.summary_received(self.summary_received)

        self.feature_client.get_summary()
        self.feature_client.get_layer_info()

    def build_info_frame(self):
        self.info_frame = pg.QtGui.QFrame()
        self.info_layout = pg.QtGui.QGridLayout()
        self.info_frame.setLayout(self.info_layout)
        self.layer_dimension_label = pg.QtGui.QLabel("Layer Output Dimensions: ")
        self.selected_filter_label = pg.QtGui.QLabel("Selected Filter: ")
        self.connection_health_label = pg.QtGui.QLabel("Server: ")
        self.fps_label = pg.QtGui.QLabel("FPS: ")



    def build_config_frame(self):
        self.config_frame = pg.QtGui.QFrame()
        self.config_layout = pg.QtGui.QGridLayout()
        self.config_frame.setLayout(self.config_layout)
        self.server_button = pg.QtGui.QPushButton("Start Server")
        self.connect_button = pg.QtGui.QPushButton("Connect")
        self.webcam_button = pg.QtGui.QPushButton("Camera")
        self.video_button = pg.QtGui.QPushButton("Video")
        self.ROI_button = pg.QtGui.QPushButton("ROI")
        self.video_slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.server_combo = QtGui.QComboBox()

        self.server_combo.addItems(list(self.settings['servers'].keys()))

        self.config_layout.addWidget(QtGui.QLabel("Server:"))
        self.config_layout.addWidget(self.server_combo, 0, 0)
        self.config_layout.addWidget(self.connect_button, 0, 1)
        self.config_layout.addWidget(self.webcam_button, 0, 2)
        self.config_layout.addWidget(self.video_button, 0, 3)
        self.config_layout.addWidget(self.ROI_button, 0, 4)
        self.config_layout.addWidget(self.server_button, 0, 5)
        self.config_layout.addWidget(self.video_slider, 1, 0, 1, 3)
        self.video_slider.sliderReleased.connect(self.frame_slider_changed)


        self.server_button.clicked.connect(self.server_clicked)
        self.webcam_button.clicked.connect(self.webcam_clicked)
        self.video_button.clicked.connect(self.video_clicked)
        self.ROI_button.clicked.connect(self.ROI_clicked)
        self.connect_button.clicked.connect(self.connect_clicked)

        return self.config_frame

    def frame_slider_changed(self):
        n_frames = self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.video_slider.value() / 99. * n_frames)


    def start_feature_server_timer(self):
        self.feature_server_timer = QtCore.QTimer()
        self.feature_server_timer.timeout.connect(self.feature_client.check)
        self.feature_server_timer.start(50)

    def connect_clicked(self):
        self.init_client(self.settings['servers'][self.server_combo.currentText()])
        if self.feature_server_timer:
            self.feature_server_timer.stop()
        self.start_feature_server_timer()


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
            row, col = divmod(i, 16)
            self.layer_layout.addWidget(button, row, col)

    def layerinfo_received(self, layerinfo):
        self.current_layer_dimensions = layerinfo['shape'][1:]
        self.current_layer_name = layerinfo['name']
        self.rows, self.cols = good_shape(self.current_layer_dimensions[-1])

    def camera_callback(self):
        if self.video_capture:
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


    def build_camera_view(self):
        self.camera_view = self.camera_window.addViewBox()
        self.camera_view.setAspectLocked(True)
        self.camera_view.invertY()
        self.camera_view.setMouseEnabled(False, False)
        self.camera_image = pg.ImageItem(border='w')
        self.camera_view.addItem(self.camera_image)

    def build_lastframe_view(self):
        self.lastframe_view = self.camera_window.addViewBox()
        self.lastframe_view.setAspectLocked(True)
        self.lastframe_view.invertY()
        self.lastframe_view.setMouseEnabled(False, False)
        self.lastframe_image = pg.ImageItem(border='w')
        self.lastframe_view.addItem(self.lastframe_image)
        self.lastframe_view.enableAutoRange()

    def build_features_view(self):
        self.features_view = self.feature_window.addViewBox()
        self.features_view.setRange(QtCore.QRectF(0, 0, 1600, 1100))
        self.features_view.setAspectLocked(True)
        self.features_view.invertY()
        self.features_image = pg.ImageItem(border='w')
        self.features_image.mouseClickEvent = self.select_filter
        self.features_image.setLevels([0, 255])
        self.features_view.addItem(self.features_image)
        self.features_view.enableAutoRange()

    def build_detailed_feature_view(self):
        self.detailed_feature_view = self.camera_window.addViewBox()
        self.detailed_feature_view.setAspectLocked(True)
        self.detailed_feature_view.invertY()
        self.detailed_features_image = pg.ImageItem(border='w')
        self.detailed_features_image.setLevels([0, 255])
        self.detailed_feature_view.addItem(self.detailed_features_image)
        self.detailed_feature_view.enableAutoRange()

    def build_views(self):
        self.build_camera_view()
        self.build_lastframe_view()
        self.build_features_view()
        self.build_detailed_feature_view()


    def select_filter(self, event):
        x = event.pos().x()
        y = event.pos().y()

        row = y // np.ceil(self.current_layer_dimensions[0] * 1.1)
        col = x // np.ceil(self.current_layer_dimensions[1] * 1.1)
        n = self.cols*row + col
        if n < self.current_layer_dimensions[2]:
            print("Selected filter {}".format(n))
            self.selected_filter = int(n)

    def prediction_received(self, result, *args, **kwargs):
        result = result['result'].squeeze()
        transposed = np.transpose(result, (2, 0, 1))
        #transposed /= np.max(transposed)
        transposed /= np.max(transposed, axis=(1, 2), keepdims=True) + 1e-3
        image = build_imagegrid(image_list=transposed, n_rows=self.rows, n_cols=self.cols)
        #image /= np.max(image)
        self.last_feature_image = np.uint8(255.0 * image)
        self.features_image.setImage(self.last_feature_image)
        self.features_view.autoRange()
        self.lastframe_view.autoRange()
        if self.selected_filter < transposed.shape[0]:
            self.detailed_features_image.setImage(255*transposed[self.selected_filter])
            self.detailed_feature_view.autoRange()
        self.do_prediction()


if __name__ == '__main__':
    import sys

    app = QtGui.QApplication([])
    z = VisualizationWindow()
    z.show()
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

