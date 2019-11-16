
import sys
import os
import glob

from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QGridLayout, QFileDialog
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QRunnable, Qt, QThreadPool  # *
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

import numba as nb
import numpy as np
import pylab as pl
from pylab import (arange, transpose, clip,  # ,scatter,show#,shape,cos,pi,reshape,dot,zeros
                   argsort, array, c_, r_,  empty, exp, float32, int32,
                   float64, hstack, int32, linspace, load, log, log10,
                   logical_and, logspace, meshgrid,  ones_like,
                   poisson, poly1d, polyfit, r_, rand, randn, ravel, real,
                   sqrt, subplots, uniform, unique, zeros, zeros_like, loadtxt, where)
import pandas as pd
import matplotlib as mpl

import lib_sinbad5

import time
import traceback

sys.path.append(os.path.abspath("./module_swig/"))
try:
    import sinbad as lib_sinbad_cpp
    lib_sinbad_cpp.initRng(-1)  # Never forget that
    print("import module sinbad OK")
except:
    print("Attention: pas de module swig sinbad")

# lg=200000
# tx=arange(lg)
# data=pl.poisson(np.sin(tx*.001)*5+20)
# prior_polya=ones_like(data)
# t0=time.time()
# for _ in range(100):
#     ret=lib_sinbad5.polya_parallel(data,1,log(3),prior_polya,1.)
# print("numba:",time.time()-t0)

# t0=time.time()
# for _ in range(100):
#     ret_cpp=lib_sinbad_cpp.polya_parallel(data.tolist(),1,log(3),prior_polya.tolist(),1.)
# print("c++:",time.time()-t0)

# pl.step(tx,data,"b",lw=.5)
# pl.step(tx,ret*sum(data),"r")
# pl.step(tx,array(ret_cpp)*sum(data),"g")
# print(sum(ret))
# pl.show()
# exit()

# from silx.gui.plot import Plot1D as silxPlot1D


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

        # print("mouseMoved", len(pts))

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
        # print("mouseDragEvent", ev)
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
    # progress = pyqtSignal(int)
    progress = pyqtSignal(int,object)

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


STOP, START, PAUSE = 0, 1, 2


class MainWindow(QMainWindow):
    """ MainWindow """

    def __init__(self, ui_file, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        loadUi(ui_file, self)
        self.show()

        # w = QWidget()

        # self.setCentralWidget(w)
        # self.area = DockArea(w)
        # gridLayout = QGridLayout(w)
        # gridLayout.setObjectName("gridLayout")
        # gridLayout.addWidget(self.area)

        # self.area comes from ui file
        d1 = Dock("full view", closable=False)
        v1 = pg.PlotWidget(title="full view")
        v1.setLogMode(x=False, y=True)
        d1.addWidget(v1)

        d2 = Dock("zoom", closable=False)
        v2 = pg.PlotWidget(title="zoom")
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
        self.spectre = array(df[0].tolist())
        tx = np.arange(len(self.spectre)+1)  # +1 because stepMode=True
        self.PlotSpectre1 = v1.plot(x=tx, y=self.spectre, pen=pg.mkPen(
            "#EA1515", width=1, style=Qt.SolidLine), name='red plot', stepMode=True)
        self.PlotSpectre2 = v2.plot(x=tx, y=self.spectre, pen=pg.mkPen(
            "#EA1515", width=1, style=Qt.SolidLine), name='zoom plot', stepMode=True)

        pos = [(854, 2), (8541, 1.8), (10000, 3)]  # rand(5,2)
        mongraph = MonGraph(v2.scene())
        mongraph.setData(pos=np.array(pos), brush=QColor(Qt.cyan),  # , text="du text"
                         pen=pg.mkPen(pg.mkColor(QColor(Qt.yellow)), width=1))  # , adj=adj, size=.2, symbol=symbols, pxMode=False, text=texts)
        v2.addItem(mongraph)

        self.regionx = pg.LinearRegionItem()
        v1.addItem(self.regionx, ignoreBounds=True)

        def updatePlot():
            self.regionx.setZValue(10)
            minX, maxX = self.regionx.getRegion()
            v2.setXRange(minX, maxX, padding=0)

        self.regionx.sigRegionChanged.connect(updatePlot)

        def updateRegion():
            self.regionx.setRegion(v2.getViewBox().viewRange()[0])

        v2.sigXRangeChanged.connect(updateRegion)
        self.regionx.setRegion([2000, 4000])

        # plot background
        xmin, xmax = v2.getViewBox().viewRange()[0]
        t_middle = .5*(tx[:-1]+tx[1:])
        test = logical_and(t_middle > xmin, t_middle < xmax)
        ext_test = r_[test, [False]]
        ext_test = np.logical_or(ext_test, np.roll(ext_test, 1))
        self.bins_computation = tx[ext_test]
        self.data_computation = self.spectre[test]
        self.PlotBackground = v2.plot(x=self.bins_computation, y=self.data_computation,
                                      pen=pg.mkPen(pg.mkColor(
                                          QColor(Qt.blue)), width=2, style=Qt.SolidLine),
                                      name='background', stepMode=True)

        def openFile():
            fname, _ = QFileDialog.getOpenFileName(self, 'Open file',
                                                   '../spectres/csv', "csv files (*.csv *.txt)")
            if fname == '':  # cancel
                return
            print(fname)
            df = pd.read_csv(fname, header=None)
            self.spectre = df[0].tolist()
            tx = np.arange(len(self.spectre)+1)
            self.PlotSpectre1.setData(x=tx, y=self.spectre)
            self.PlotSpectre2.setData(x=tx, y=self.spectre)

        self.openButton.clicked.connect(openFile)
        self.actionopen.triggered.connect(openFile)

        # self.start_compute_thread()
        # compute thread
        self.threadpool = QThreadPool()
        print("Multithreading with maximum %d threads" %
              self.threadpool.maxThreadCount())
        self.state_computation = STOP
        self.pushButtonStartPause.clicked.connect(self.StartPauseCompute)
        self.pushButtonStop.clicked.connect(self.StopCompute)

    def StopCompute(self):
        # -> STOP
        self.state_computation = STOP
        self.pushButtonStartPause.setChecked(False)

    def StartPauseCompute(self):
        if self.state_computation == START:  # -> PAUSE
            self.state_computation = PAUSE

        elif self.state_computation == STOP:  # -> START + new computation
            self.start_compute_thread()
            self.state_computation = START

        elif self.state_computation == PAUSE:  # -> START
            self.state_computation = START

        else:
            pass

    def progress_fn(self, n_iterations, background):
        # self.bar.setValue(n)
        self.PlotBackground.setData(self.bins_computation, background)
        print(n_iterations, time.time())

    def GibbsSampler(self, progress_callback, paramGibbs):
        # -----------------------------------------

        print(len(paramGibbs))
        ofst = paramGibbs['ofst']
        kevcan = paramGibbs["kevcan"]
        var1 = paramGibbs["var1"]
        var0 = paramGibbs["var0"]
        alpha = paramGibbs["alpha"]
        m0 = paramGibbs["m0"]
        h0 = paramGibbs["h0"]
        excluy = paramGibbs["excluy"]
        exclux = paramGibbs["exclux"]
        probaExclu = paramGibbs["probaExclu"]
        fit = 1  # paramGibbs["fit"]
        ppolya = paramGibbs["ppolya"]
        prior_var = paramGibbs["prior_var"]
        typenoy = paramGibbs["typenoy"]
        seuilpval = 15.  # paramGibbs["seuilpval"]#20
        prec = paramGibbs["prec"]  # 0.975
        typed = paramGibbs["typed"]  # 1
        ppriorplatvspr = paramGibbs["ppriorplatvspr"]
        energmin = paramGibbs["emin"]
        energmax = paramGibbs["emax"]

        # -----------------------------------------
        # energmin=600.
        # energmax=900.
        # ppolya=ones(numcans)
        print("energmin energmax", energmin, energmax)

        canmin = int32((energmin-ofst)/kevcan)
        canmax = int32((energmax-ofst)/kevcan)
        numcans = canmax-canmin
        energmin = (canmin-.0)*kevcan+ofst
        energmax = energmin+numcans*kevcan
        binobs = r_[0:numcans+1]*kevcan+energmin
        energquant = .5*(binobs[:-1]+binobs[1:])
        # pour essai
        rep = "../fred/21 mars 2016 - 32 spectres 134Cs et 137Cs/"
        data = np.loadtxt(glob.glob(rep+"*[0-9][0-9].txt")[0])
        histo = array(data[canmin:canmax], 'f')
        # lib_sinbad_cpp.initialisation(
        #     binobs.tolist(), histo.astype('l').tolist(), var1, var0)

        # @nb.jit(nogil=True, parallel=False)
        # def appel_iter(i):
        #     ret = array(sinbad.appel_iter(var1, var0, (iter == 0) | (iter == 3), alpha, m0,
        #                                   h0, excluy, exclux, probaExclu, fit, ppolya, prior_var, typenoy, ppriorplatvspr))
        #     return i

        numthreads = self.threadpool.maxThreadCount()
        print("numthreads", numthreads)
        data = array(self.data_computation, dtype=int)
        prior_polya = ones_like(data)

        while True:
            ni = 0
            t0 = time.time()
            while time.time()-t0 < .5:
                if self.state_computation == STOP:
                    break
                while self.state_computation == PAUSE:
                    time.sleep(.25)
                ret = lib_sinbad5.polya_parallel(data, 1, log(3),
                                                 prior_polya, 1.)
                
                # ret = array(lib_sinbad_cpp.polya_parallel(data.tolist(), 1, log(3),
                #                                   prior_polya.tolist(), 1.))
                ni += 1
            ret *= sum(data)    
            progress_callback.emit(ni, ret)
            if self.state_computation == STOP:
                break

        return "Done."

    def print_output(self, s):
        print("final result:", s)

    def thread_complete(self):
        print("END COMPUTATION")

    def start_compute_thread(self):
        # Pass the function to execute
        # Any other args, kwargs are passed to the run function
        paramGibbs = np.load("dicoFred.npy", allow_pickle=True,
                             encoding='latin1').tolist()
        worker = Worker(self.GibbsSampler, paramGibbs=paramGibbs)

        worker.signals.result.connect(self.print_output)
        worker.signals.finished.connect(self.thread_complete)
        worker.signals.progress.connect(self.progress_fn)

        # Execute
        self.threadpool.start(worker)


print(sys.version)
# print("QT VERSION:", PySide2Version)
print("matplotlib VERSION:", mpl.__version__)
print("pyqtgraph version", pg.__version__)
print("numba version", nb.__version__)
app = QApplication(sys.argv)
window = MainWindow("sinbad5.ui")
# window = MainWindow2("sinbad5.ui")
app.exec_()

# app = QApplication([])
# window = MainWindow()
# app.exec_()
