from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import pyqtgraph as pg

app = QtGui.QApplication([])
import cv2
from remote import FeatureClient

cap = cv2.VideoCapture(0)

def good_shape(n, min_aspect=1.0, max_aspect=4.0, max_width=np.inf, max_height=np.inf, min_width=1, min_height=1, method='this is only here for backwards compatibility'):
    from numpy import log
    import scipy.optimize
    import itertools
    c = np.float32([1, .1]) # 0.1*logW + log_H
    A_ub = np.float32([[-1, -1],
                       [-1, 1],
                       [1, -1]])
    b = np.float32([-log(n), log(max_aspect), -log(min_aspect)])
    bounds = ((log(min_height), log(max_height)), (log(min_width), log(max_width)))
    res = scipy.optimize.linprog(c, A_ub=A_ub, b_ub=b, bounds=bounds)
    H, W = np.exp(res.x).tolist()
    candidates = []
    for op_a, op_bb in itertools.product([np.ceil, np.floor], [np.ceil, np.floor]):
        W_p = op_a(W)
        H_p = op_bb(H)
        if W_p*H_p >= n: # we don't really check all the constraints for feasibility, just the area constraint
            candidates.append((H_p, (int(H_p), int(W_p)))) # Heuristic: accept the shortest (sort-of) feasible solution

    candidates = sorted(candidates)
    #assert len(candidates) > 0
    return candidates[0][1]

class VisualizationWindow(QtGui.QMainWindow):
    def __init__(self):
        super(VisualizationWindow, self).__init__()
        self.win = pg.GraphicsLayoutWidget(self)
        self.setCentralWidget(self.win)
        self.setGeometry(0, 200, 1600, 900)
        self.selected_filter = 1

        self.current_layer_dimensions = (512, 7, 7)
        self.ready = True

        self.rows, self.cols = good_shape(self.current_layer_dimensions[0])
        self.feature_client = FeatureClient()
        self.feature_client.prediction_received(self.prediction_received)
        self.video_capture = cv2.VideoCapture(0)
        self.last_frame = None
        self.rois = []

        self.build_views()
        self.build_feature_grid()
        self.start_timers()

    def camera_callback(self):
        # data = np.random.normal(size=(15, 600, 600), loc=1024, scale=64).astype(np.uint16)
        has_frame, frame = self.video_capture.read()
        if has_frame:
            ## Display the data
            self.last_frame = frame
            self.camera_image.setImage(np.transpose(frame[:, :, ::-1], (1, 0, 2)))

    def do_prediction(self):
        if np.any(self.last_frame) and self.ready:
            self.feature_client.predict(self.last_frame)

    def start_timers(self):
        self.camera_timer = QtCore.QTimer()
        self.camera_timer.timeout.connect(self.camera_callback)
        self.camera_timer.start(10)

        self.feature_server_timer = QtCore.QTimer()
        self.feature_server_timer.timeout.connect(self.feature_client.check)
        self.feature_server_timer.start(50)

        self.prediction_timer = QtCore.QTimer()
        self.prediction_timer.timeout.connect(self.do_prediction)
        self.prediction_timer.start(200)

    def build_views(self):
        self.camera_view = self.win.addViewBox(row=0, col=0, rowspan=1, colspan=1)
        self.layers_view = self.win.addViewBox(row=1, col=0, rowspan=1, colspan=4)
        self.features_view = self.win.addViewBox(row=2, col=0, rowspan=4, colspan=4)
        self.detailed_feature_view = self.win.addViewBox(row=0, col=1, rowspan=1, colspan=1)

        self.camera_view.setAspectLocked(True)
        self.features_view.setAspectLocked(True)
        self.detailed_feature_view.setAspectLocked(True)
        # view2.setMovable()
        self.camera_view.invertY()
        self.detailed_feature_view.invertY()
        self.features_view.invertY()

        self.camera_image = pg.ImageItem(border='w')
        self.features_image = pg.ImageItem(border='w')
        self.detailed_features_image = pg.ImageItem(border='w')
        self.features_image.setLevels([0, 255])
        self.detailed_features_image.setLevels([0, 255])
        self.camera_view.addItem(self.camera_image)
        self.features_view.addItem(self.features_image)
        self.detailed_feature_view.addItem(self.detailed_features_image)

        self.features_view.enableAutoRange()
        self.detailed_feature_view.enableAutoRange()

    def set_selected(self, roi, event):
        self.selected_filter = roi.row * self.cols + roi.col

    def build_feature_grid(self):
        n, w, h = self.current_layer_dimensions
        for i in range(self.rows):
            for j in range(self.cols):
                r = pg.ROI((j * w, i * w), size=(w, h), movable=False, removable=False)
                r.row = i
                r.col = j
                r.setAcceptedMouseButtons(QtCore.Qt.LeftButton)
                r.sigClicked.connect(self.set_selected)
                self.rois.append(r)
                self.features_view.addItem(r)

    def prediction_received(self, result, *args, **kwargs):
        result = result['result'].squeeze()
        transposed = np.transpose(result, (2, 0, 1))

        num_filters, width, height = transposed.shape
        image = np.zeros((self.rows * width, self.cols * height))
        side = transposed.shape[1]
        for row in range(self.rows):
            for col in range(self.cols):
                if row * self.cols + col >= num_filters:
                    break
                image[row * side:(row + 1) * side, col * side:(col + 1) * side] = transposed[row * self.cols + col]
        image /= np.max(image)
        image = np.uint8(255.0 * image)
        self.features_image.setImage(image.T)

        if self.selected_filter:
            zzz = transposed[self.selected_filter].squeeze()
            zzz /= (np.max(zzz) + 1e-5)
            self.detailed_features_image.setImage(zzz.T)


z = VisualizationWindow()
win = z.win
z.show()

if __name__ == '__main__':
    import sys

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        print("let's go")
        QtGui.QApplication.instance().exec_()
