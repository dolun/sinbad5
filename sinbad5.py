
import sys

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.uic import loadUi

# from PySide2 import __version__ as PySide2Version
# from PySide2.QtGui import *
# from PySide2.QtWidgets import *
# from PySide2.QtCore import *
# from PySide2.QtUiTools import QUiLoader
# from PySide2.QtWebEngineWidgets import QWebEngineView

import pyqtgraph as pg
# from pyqtgraph.console import ConsoleWidget
from pyqtgraph.dockarea import Dock, DockArea

import numpy as np
import pylab as pl
import pandas as pd
import matplotlib as mpl
import time
import traceback

from silx.gui.plot import Plot1D as silxPlot1D


class MonGraph(pg.GraphItem):
    def __init__(self, scene):
        self.dragPoint = None
        self.dragOffset = None
        self.textItems = []
        pg.GraphItem.__init__(self)
        scene.sigMouseMoved.connect(self.mouseMoved)

    def mouseMoved(self, evt):

        vb = self.getViewBox()
        mousePoint = vb.mapSceneToView(evt)
        pts = self.scatter.pointsAt(mousePoint)
        if len(pts):
            self.setCursor(Qt.PointingHandCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

        print("mouseMoved", len(pts))

    def setData(self, **kwds):
        print("setData", kwds)
        self.text = kwds.pop('text', [])
        self.data = kwds
        if 'pos' in self.data:
            npts = self.data['pos'].shape[0]
            self.data['data'] = np.empty(npts, dtype=[('index', int)])
            self.data['data']['index'] = np.arange(npts)
        else:
            print("pas de pos!!")
            return
        # self.setTexts(self.text)
        self.updateGraph()

    # def setTexts(self, text):
    #     for i in self.textItems:
    #         i.scene().removeItem(i)
    #     self.textItems = []
    #     for t in text:
    #         item = pg.TextItem(t)
    #         self.textItems.append(item)
    #         item.setParentItem(self)

    def updateGraph(self):
        srt = np.argsort(self.data['pos'][:, 0])
        self.data['adj'] = np.c_[srt[:-1], srt[1:]]
        # print("updateGraph")
        pg.GraphItem.setData(self, **self.data)
        # for i,item in enumerate(self.textItems):
        # item.setPos(*self.data['pos'][i])
    # def HoverEvent(self, ev):
        # print("HoverEvent",ev)

    def wheelEvent(self, ev):
        print("wheelEvent", ev.delta())
        ev.ignore()

    def mouseClickEvent(self, ev):
        print("mouseClickEvent", ev)
        ev.ignore()
        # if ev.button()==2:
        # print(ev.buttonDownPos())
        # print("suppression!!")
        # ev.accept()
        # else:
        # ev.ignore()

    def mouseReleaseEvent(self, ev):
        print("mouseReleaseEvent")

    def mousePressEvent(self, ev):
        pos = ev.buttonDownPos(pg.QtCore.Qt.LeftButton)
        pts = self.scatter.pointsAt(pos)
        print("mousePressEvent", pos, ev.button(), len(pts))
        if len(pts) == 0 or ev.button() != pg.QtCore.Qt.RightButton:  # 2:
            ev.ignore()
            return
        ind = pts[0].data()[0]

        self.data['pos'] = np.r_[self.data['pos']
                                 [:ind], self.data['pos'][ind+1:]]
        # self.data['brush']=r_[self.data['brush'][:ind],self.data['brush'][ind+1:]]

        # self.data['brush'][ind]=Qt.blue#QColor("#219221")
        self.setData(pos=self.data['pos'])  # , brush=self.data['brush'])
        ev.accept()

    def mouseDoubleClickEvent(self, ev):
        pos = ev.pos()
        print("mouseDoubleClickEvent", ev, [pos.x(), pos.y()])
        self.data['pos'] = np.r_[self.data['pos'], [[pos.x(), pos.y()]]]
        # self.data['brush']=r_[self.data['brush'],[Qt.blue]]
        self.setData(pos=self.data['pos'])
        # ev.accept()
        ev.ignore()

    def mouseMoveEvent(self, ev):
        print("mouseMoveEvent"), ev
        ev.ignore()

    def mouseDragEvent(self, ev):
        print("mouseDragEvent", ev)
        if ev.button() != pg.QtCore.Qt.LeftButton:
            ev.ignore()
            return

        if ev.isStart():
            # We are already one step into the drag.
            # Find the point(s) at the mouse cursor when the button was first
            # pressed:
            pos = ev.buttonDownPos()
            pts = self.scatter.pointsAt(pos)
            print("***", pts)
            if len(pts) == 0:
                ev.ignore()
                return
            self.dragPoint = pts[0]
            ind = pts[0].data()[0]
            print(pts[0].data())
            self.dragOffset = self.data['pos'][ind] - pos
        elif ev.isFinish():
            ind = self.dragPoint.data()[0]
            # self.data['brush'][ind]=QColor("#FF0000")
            self.dragPoint = None
            self.updateGraph()
            return
        else:
            if self.dragPoint is None:
                ev.ignore()
                return

        ind = self.dragPoint.data()[0]
        newpt = ev.pos() + self.dragOffset
        # newpt[0] = np.clip(newpt[0], 0, 1)
        self.data['pos'][ind] = newpt
        # print(ev.pos(),self.dragOffset)
        self.updateGraph()
        ev.accept()

    def clicked(self, pts): print("clicked: %s" % pts)


class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data

    error
        `tuple` (exctype, value, traceback.format_exc() )

    result
        `object` data returned from processing, anything

    progress
        `int` indicating % progress 

    '''
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)
    # finished = Signal()
    # error = Signal(tuple)
    # result = Signal(object)
    # progress = Signal(int)


class Worker(QRunnable):

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the callback to our kwargs
        self.kwargs['progress_callback'] = self.signals.progress

    @pyqtSlot()
    # @Slot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''

        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
            # print("------------------")
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            # Return the result of the processing
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()  # Done


class MainWindow(QMainWindow):
    """ MainWindow """

    def __init__(self, ui_file, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        loadUi(ui_file, self)
        self.show()

        w = QWidget()

        self.setCentralWidget(w)
        self.area = DockArea(w)
        gridLayout = QGridLayout(w)
        gridLayout.setObjectName("gridLayout")
        gridLayout.addWidget(self.area)

        d1 = Dock("full view", closable=False)
        v1 = pg.PlotWidget(title="vue 1")
        v1.setLogMode(x=False, y=True)
        d1.addWidget(v1)

        d2 = Dock("zoom", closable=False)
        v2 = pg.PlotWidget(title="vue 2")
        v2.setLogMode(x=False, y=True)
        d2.addWidget(v2)

        # d3 = Dock("silx", closable=False)
        # plotsilx = silxPlot1D()  # Create the plot widget
        # plotsilx.addCurve(x=(1, 2, 3), y=(1.5, 2, 1), legend='curve')
        # d3.addWidget(plotsilx)

        self.area.addDock(d1)
        self.area.addDock(d2, 'bottom')
        # self.area.addDock(d3, 'right')


        df = pd.read_csv("../spectres/csv/pechblend.csv", header=None)
        spectre = df[0].tolist()
        tx = np.arange(len(spectre)+1) # +1 because stepMode=True
        self.spectre1 = v1.plot(x=tx, y=spectre, pen=pg.mkPen(
            "#EA1515", width=1, style=Qt.SolidLine), name='red plot', stepMode=True)
        self.spectre2 = v2.plot(x=tx, y=spectre, pen=pg.mkPen(
            "#EA1515", width=1, style=Qt.SolidLine), name='zoom plot', stepMode=True)

        mongraph = MonGraph(v1.scene())
        pos = [(854, 2), (8541, 1.8), (10000, 3)]  # rand(5,2)
        mongraph.setData(pos=np.array(pos), brush=QColor(Qt.cyan),  # , text="du text"
                         pen=pg.mkPen(pg.mkColor("#FFFF00"), width=2))  # , adj=adj, size=.2, symbol=symbols, pxMode=False, text=texts)
        v2.addItem(mongraph)

        self.regionx = pg.LinearRegionItem()
        v1.addItem(self.regionx, ignoreBounds=True)
        self.regionx.setRegion([2000, 4000])

        def updateRegion():
            self.regionx.setZValue(10)
            minX, maxX = self.regionx.getRegion()
            v2.setXRange(minX, maxX, padding=0)

        self.regionx.sigRegionChanged.connect(updateRegion)

        def openFile():
            fname, _ = QFileDialog.getOpenFileName(self, 'Open file',
                                                   '../spectres/csv', "csv files (*.csv *.txt)")
            print(fname)
            df = pd.read_csv(fname, header=None)
            spectre = df[0].tolist()
            tx = np.arange(len(spectre)+1)
            self.spectre1.setData(x=tx, y=spectre)
            self.spectre2.setData(x=tx, y=spectre)

        self.openButton.clicked.connect(openFile)

#############################################################################
        return

        w = QWidget()

        self.setCentralWidget(w)

        self.counter = 0

        layout = QVBoxLayout()
        self.bar = QProgressBar()

        self.l = QLabel("Start")
        b = QPushButton("DANGER!")
        b.pressed.connect(self.oh_no)

        # self.browser = QWebEngineView()
        # self.browser.setUrl(QUrl("http://google.com"))
        self.area = DockArea(w)

        d1 = Dock("vp", closable=False)
        d2 = Dock("vz", closable=False)
        v1 = pg.PlotWidget(title="vue 1")
        d1.addWidget(v1)
        v2 = pg.PlotWidget(title="vue 2")
        d2.addWidget(v2)
        self.area.addDock(d1, 'left')
        self.area.addDock(d2, 'right')
        mongraph = MonGraph(v1.scene())
        pos = [(0, 1), (.5, 0), (.75, 0.8), (1, 1)]  # rand(5,2)
        mongraph.setData(pos=np.array(pos), brush=QColor("#FF0050"),  # , text="du text"
                         pen=pg.mkPen(pg.mkColor("#FFFF00"), width=2))  # , adj=adj, size=.2, symbol=symbols, pxMode=False, text=texts)
        v1.addItem(mongraph)

        layout.addWidget(self.l)
        layout.addWidget(b)
        layout.addWidget(self.bar)
        # layout.addWidget(self.browser)
        layout.addWidget(self.area)
        w.setLayout(layout)
        # QFileDialog.getOpenFileName(self,'Open file')
        # gridLayout = QGridLayout()
        # gridLayout.setObjectName("gridLayout")
        # gridLayout.addWidget(self.area)

        self.show()

        self.threadpool = QThreadPool()
        print("Multithreading with maximum %d threads" %
              self.threadpool.maxThreadCount())

        self.timer = QTimer()
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.recurring_timer)
        self.timer.start()

    def progress_fn(self, n):
        self.bar.setValue(n)
        print(f"{n}% done")

    def execute_this_fn(self, progress_callback):
        for n in range(0, 5):
            time.sleep(1)
            progress_callback.emit(n*100/4)

        return "Done."

    def print_output(self, s):
        print("resultat:", s)

    def thread_complete(self):
        print("THREAD COMPLETE!")

    def oh_no(self):
        # Pass the function to execute
        # Any other args, kwargs are passed to the run function
        worker = Worker(self.execute_this_fn)
        worker.signals.result.connect(self.print_output)
        worker.signals.finished.connect(self.thread_complete)
        worker.signals.progress.connect(self.progress_fn)

        # Execute
        self.threadpool.start(worker)

    def recurring_timer(self):
        self.counter += 1
        self.l.setText("Counter: %d" % self.counter)


print(sys.version)
# print("QT VERSION:", PySide2Version)
print("matplotlib VERSION:", mpl.__version__)
print("pyqtgraph version", pg.__version__)
app = QApplication(sys.argv)
window = MainWindow("sinbad5.ui")
# window = MainWindow2("sinbad5.ui")
app.exec_()

# app = QApplication([])
# window = MainWindow()
# app.exec_()
