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
        self.win = pg.GraphicsLayoutWidget(self)
        self.setCentralWidget(self.win)
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

        self.build_views()
        self.start_timers()

    def set_selected_layer(self, layer=None, button=None):
        if button:
            if self.last_button:
                self.last_button.setChecked(False)
            button.setChecked(True)
            self.last_button = button
        self.feature_client.change_layer(layer)
        self.feature_client.get_layer_info()

    def summary_received(self, summary):
        self.layer_info = summary['result']

        proxy = pg.QtGui.QGraphicsProxyWidget()
        frame = pg.QtGui.QFrame()
        p = frame.palette()
        p.setColor(frame.backgroundRole(), pg.QtGui.QColor("black"))
        frame.setPalette(p)

        frame.setGeometry(0, 0, 1, 1)
        layout = pg.QtGui.QHBoxLayout()
        frame.setLayout(layout)
        proxy.setWidget(frame)

        for i, layer in enumerate(self.layer_info):
            button = pg.QtGui.QPushButton("{}".format(layer))
            button.setCheckable(True)
            button.clicked.connect(functools.partial(self.set_selected_layer, layer=i, button=button))
            layout.addWidget(button)
        self.layers_view.addItem(proxy)

        self.layers_view.setRange(QtCore.QRectF(0, 0, 1600, 100))

    def layerinfo_received(self, layerinfo):
        self.current_layer_dimensions = layerinfo['shape'][1:]
        self.selector.setSize(self.current_layer_dimensions[1], self.current_layer_dimensions[2])
        #self.selector.snapSize = self.current_layer_dimensions[2]
        self.current_layer_name = layerinfo['name']
        self.rows, self.cols = good_shape(self.current_layer_dimensions[-1])
        self.selector.setPos((0, 0))




    def camera_callback(self):
        has_frame, frame = self.video_capture.read()
        first = False
        if has_frame:
            ## Display the data
            if not np.any(self.last_frame):
                first = True
            self.last_frame = frame
            self.camera_image.setImage(frame[:, :, ::-1])
        if first:
            self.do_prediction()

    def do_prediction(self):
        if np.any(self.last_frame) and self.ready:
            self.lastframe_image.setImage(self.last_frame[:,:,::-1])
            self.feature_client.predict(self.last_frame[:, :, ::-1])

    def start_timers(self):
        self.camera_timer = QtCore.QTimer()
        self.camera_timer.timeout.connect(self.camera_callback)
        self.camera_timer.start(10)

        self.feature_server_timer = QtCore.QTimer()
        self.feature_server_timer.timeout.connect(self.feature_client.check)
        self.feature_server_timer.start(50)

        #self.prediction_timer = QtCore.QTimer()
        #self.prediction_timer.timeout.connect(self.do_prediction)
        #self.prediction_timer.start(100)

    def build_views(self):
        self.camera_view = self.win.addViewBox(row=0, col=0, rowspan=1, colspan=1)
        self.lastframe_view = self.win.addViewBox(row=0, col=1, rowspan=1, colspan=1)
        self.layers_view = self.win.addViewBox(row=1, col=0, rowspan=1, colspan=3)
        self.features_view = self.win.addViewBox(row=2, col=0, rowspan=8, colspan=3)
        self.detailed_feature_view = self.win.addViewBox(row=0, col=2, rowspan=1, colspan=1)
        self.features_view.mouseClickEvent = self.select_filter
        self.camera_view.setAspectLocked(True)
        self.features_view.setAspectLocked(True)
        self.layers_view.setAspectLocked(True)
        self.detailed_feature_view.setAspectLocked(True)
        self.lastframe_view.setAspectLocked(True)

        # view2.setMovable()
        self.camera_view.invertY()
        self.detailed_feature_view.invertY()
        self.features_view.setMouseEnabled(False, False)
        self.layers_view.invertY()
        self.layers_view.setMouseEnabled(False, False)
        self.features_view.invertY()
        self.lastframe_view.invertY()
        self.lastframe_view.setMouseEnabled(False, False)



        self.camera_image = pg.ImageItem(border='w')
        self.lastframe_image = pg.ImageItem(border='w')
        self.features_image = pg.ImageItem(border='w')
        self.detailed_features_image = pg.ImageItem(border='w')
        self.selector = pg.ROI((0,0), size=(self.current_layer_dimensions[1],self.current_layer_dimensions[2]))
        self.selector.setAcceptedMouseButtons(QtCore.Qt.LeftButton)
        self.selector.sigRegionChanged.connect(self.selector_moved)
        self.selector.sigRegionChangeFinished.connect(self.selector_mouseup)

        self.features_view.addItem(self.selector)

        self.features_image.setLevels([0, 255])
        self.detailed_features_image.setLevels([0, 255])

        self.camera_view.addItem(self.camera_image)
        self.features_view.addItem(self.features_image)
        self.detailed_feature_view.addItem(self.detailed_features_image)
        self.lastframe_view.addItem(self.lastframe_image)

        self.features_view.enableAutoRange()
        self.layers_view.enableAutoRange()
        self.detailed_feature_view.enableAutoRange()
        self.lastframe_view.enableAutoRange()

    def select_filter(self, event):
        pos_item = self.features_view.mapFromViewToItem(self.features_image, QtCore.QPointF(event.pos().x(), event.pos().y()))
        x = pos_item.x()
        y = pos_item.y()
        x_size, y_size = self.features_view.viewPixelSize()
        print('lastfeatureimageshape', self.last_feature_image.shape)
        print('size', x_size, y_size)
        print(x, y)
        x /= x_size
        y /= y_size

        print(x, y)

        h, w = self.current_layer_dimensions[:-1]
        x /= w
        y /= h
        print(x,y)
        print('----')


    def selector_moved(self, roi=None, event=None):
        if np.any(self.last_feature_image):
            self.detailed_features_image.setImage(self.selector.getArrayRegion(self.last_feature_image, self.features_image))

    def selector_mouseup(self, *args, **kwargs):
        x = np.round(self.selector.pos().x()/self.current_layer_dimensions[0])*self.current_layer_dimensions[0]
        y = np.round(self.selector.pos().y()/self.current_layer_dimensions[1])*self.current_layer_dimensions[1]
        self.selector.setPos((x, y), update=False)


    def prediction_received(self, result, *args, **kwargs):
        result = result['result'].squeeze()
        transposed = np.transpose(result, (2, 0, 1))
        image = build_imagegrid(image_list=transposed, n_rows=self.rows, n_cols=self.cols)
        image /= np.max(image)
        self.last_feature_image = np.uint8(255.0 * image)
        self.features_image.setImage(self.last_feature_image)
        self.features_view.autoRange()
        self.lastframe_view.autoRange()
        self.detailed_features_image.setImage(self.selector.getArrayRegion(self.last_feature_image, self.features_image))
        self.detailed_feature_view.autoRange()
        self.do_prediction()


if __name__ == '__main__':
    import sys

    app = QtGui.QApplication([])
    z = VisualizationWindow()
    win = z.win
    z.show()
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

