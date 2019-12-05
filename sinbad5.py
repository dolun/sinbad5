
import sys
import os
import glob

from PyQt5.QtGui import QColor, QBrush
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
                   float64, hstack, vstack, linspace, load, log, log10,
                   logical_and, logspace, meshgrid,  ones_like,
                   poisson, poly1d, polyfit, r_, rand, randn, ravel, real,
                   sqrt, subplots, uniform, unique, zeros, zeros_like, loadtxt, where)
import pandas as pd
import matplotlib as mpl

import lib_sinbad5

import time
import traceback

# @nb.njit(nb.int32[:](nb.int32[:]),parallel=True)
# def trynb(ret):
#     n=len(ret)
#     for _ in nb.prange(1000000):
#         p=rand(n)
#         p/=np.sum(p)
#         ret+=pl.multinomial(1000,p)
#     return ret

# t=zeros(1000,dtype=int32)
# retour=trynb(t)
# print(sum(retour))
# exit()
########

sys.path.append(os.path.abspath("./module_swig/"))
try:
    import sinbad as lib_sinbad_cpp
    lib_sinbad_cpp.initRng(-1)  # Never forget that
    print("import module sinbad OK")
except:
    print("Attention: pas de module swig sinbad")
'''
t0=time.time()
ret=array(lib_sinbad_cpp.draw_from_multinomial(
    10000,100000))
print(time.time()-t0)
print(ret.sum())
print("-------------")


@nb.njit(nb.int32[:](nb.int32,nb.int32),nogil=True, parallel=False)
def ma_multi(s,n):
    l=100
    p=np.ones(l)/l
    ret=zeros_like(p,dtype=np.int32)
    for _ in range(n):
        # ret=pl.multinomial(s,p)
        ret+=lib_sinbad5.multinomial_knuth(s,p)
    return ret

@nb.njit(nb.int32(nb.int32,nb.int32),nogil=True, parallel=False)
def ma_binom(s,n):
    ret=0
    for _ in range(n):
        # ret+=pl.binomial(s,.5)
        ret+=lib_sinbad5.binomial_knuth(s,.5)
    return ret

t0=time.time()
ret=array(ma_multi(100000,10000))
print(time.time()-t0)
print(ret.sum())

exit()
'''
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
        if vb is None:
            return
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
    progress = pyqtSignal(int, object)

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
        self.setWindowTitle("-SINBAD-")
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

        # désactive l'interactivité des axes
        v1.setMouseEnabled(x=False, y=False)
        v1.showGrid(x=True, y=True)
        v1.setBackgroundBrush(QBrush(QColor("#FBFFEB")))
        # propriétés CSS à utiliser pour le label
        labelStyle = {'color': '#00EEBB', 'font-size': '12pt'}
        v1.getAxis('bottom').setLabel(
            'energy', units='kev', **labelStyle)  # label de l'axe
        v1.getAxis('left').setLabel('I.', units='cps',
                                    **labelStyle)  # label de l'axe

        d2 = Dock("zoom", closable=False)
        self.zoom_widget = pg.PlotWidget(title="zoom")
        vz = self.zoom_widget
        viewBoxz = vz.getPlotItem().getViewBox()
        vz.setLogMode(x=False, y=True)
        # print("vz.listDataItems()",self.zoom_widget.getAxis('left').logMode)
        vz.setMouseEnabled(x=True, y=True)  # désactive interactivité axe X
        vz.showGrid(x=True, y=True)
        vz.setBackgroundBrush(QBrush(QColor(Qt.black)))
        d2.addWidget(self.zoom_widget)

        d3 = Dock("results", closable=False)
        self.results_widget = pg.DataTreeWidget()
        d3.setMaximumWidth(200)
        d3.addWidget(self.results_widget)
        d3.setTitle("-results-")

        # d3 = Dock("silx", closable=False)
        # plotsilx = silxPlot1D()  # Create the plot widget
        # plotsilx.addCurve(x=(1, 2, 3), y=(1.5, 2, 1), legend='curve')
        # d3.addWidget(plotsilx)

        self.area.addDock(d1)
        self.area.addDock(d2, 'bottom')
        self.area.addDock(d3, 'right')

        def openFile():
            fname, _ = QFileDialog.getOpenFileName(
                self, 'Open file', '../spectres/csv', "csv files (*.csv *.txt)")
            if fname != '':
                set_new_data(fname)

        def set_new_data(file_data):
            print(file_data)
            df = pd.read_csv(file_data, header=None)
            self.spectre = array(df[0].tolist())
            self.bins_spectre = np.arange(len(self.spectre)+1)*.25763
            self.PlotSpectre1.setData(x=self.bins_spectre, y=self.spectre)
            self.PlotSpectre2.setData(x=self.bins_spectre, y=self.spectre)

        spectrumColor = Qt.green
        self.PlotSpectre1 = v1.plot(x=[0, 1], y=[1], pen=pg.mkPen(
            pg.mkColor(QColor(spectrumColor)), width=1, style=Qt.SolidLine), name='red plot', stepMode=True)
        self.PlotSpectre2 = self.zoom_widget.plot(x=[0, 1], y=[1], pen=pg.mkPen(
            pg.mkColor(QColor(spectrumColor)), width=1, style=Qt.SolidLine), name='zoom plot', stepMode=True)
        set_new_data("./pechblend.csv")

        self.regionx = pg.LinearRegionItem()
        v1.addItem(self.regionx, ignoreBounds=True)

        def updatePlot():
            self.regionx.setZValue(10)
            minX, maxX = self.regionx.getRegion()
            vz.setXRange(minX, maxX, padding=0)

        self.regionx.sigRegionChanged.connect(updatePlot)

        def updateRegion():  
            self.regionx.setRegion(
                vz.getViewBox().viewRange()[0])
            # stateVBZ = viewBoxz.getState()
            # _, isAutoRangeY = stateVBZ["autoRange"]
            # self.actionAutoRangeY.setChecked(isAutoRangeY)

        vz.sigXRangeChanged.connect(updateRegion)
        self.regionx.setRegion([200, 800])

        # plot background
        colorPlotReconstruction = QColor(Qt.red)
        self.PlotReconstruction = vz.plot(x=self.bins_spectre[:2], y=self.spectre[:1],
                                          pen=pg.mkPen(pg.mkColor(
                                              colorPlotReconstruction), width=2,
            style=Qt.SolidLine),
            name='background', stepMode=True)
        self.PlotReconstruction.setOpacity(.01 *
                                           self.sliderOpacityReconstruction.value())
        self.PlotBackground = vz.plot(x=self.bins_spectre[:2], y=self.spectre[:1],
                                      pen=pg.mkPen(pg.mkColor(
                                          QColor(Qt.blue)), width=1, style=Qt.SolidLine),
                                      name='background', stepMode=True)

        # plot pics
        print("self.PlotBackground.pos", self.PlotBackground.y())

        self.PlotPics = pg.GraphItem()
        vz.addItem(self.PlotPics)
        # self.PlotPics.setData(**getDataForPlotPics((300, 400., 600), (5, 6, 4.5)))
        #  manual_polya_prior_plot
        self.manualPolyaPriorPlot = MonGraph(vz.scene())
        vz.addItem(self.manualPolyaPriorPlot)
        self.set_data_computation()

        # actions
        self.actionOpen.triggered.connect(openFile)

        def setAutoRangeY(val):
            vz.setMouseEnabled(x=True, y=not val)
            if val:
                vz.enableAutoRange(axis="y")
                vz.setAutoVisible(y=val)
            else:
                vz.disableAutoRange(axis="y")

        self.actionAutoRangeY.triggered.connect(setAutoRangeY)

        def setOpacityPlotReconstruction(val):
            # colorPlotReconstruction = QColor(Qt.red)
            # colorPlotReconstruction.setAlphaF(.01*val)
            # self.PlotReconstruction.setPen(pg.mkPen(pg.mkColor(
            #     colorPlotReconstruction), width=3,
            #     style=Qt.SolidLine))
            self.PlotReconstruction.setOpacity(.01*val)

        self.sliderOpacityReconstruction.valueChanged.connect(
            setOpacityPlotReconstruction)
        # print(vz.getPlotItem().getViewBox().getState())
        # self.start_compute_thread()
        # compute thread
        self.threadpool = QThreadPool()
        print("Multithreading with maximum %d threads" %
              self.threadpool.maxThreadCount())
        self.state_computation = STOP
        self.actionStartPause.triggered.connect(self.StartPauseCompute)
        self.actionStop.triggered.connect(self.StopCompute)
        # buttons

        def setVisiblePriorPolya(val):
            if val:
                vz.addItem(self.manualPolyaPriorPlot)
            else:
                vz.removeItem(self.manualPolyaPriorPlot)

        self.checkBoxVisibleManualPriorPolya.toggled.connect(setVisiblePriorPolya)

    def set_data_computation(self):
        xmin, xmax = self.zoom_widget.getViewBox().viewRange()[0]
        t_middle = .5*(self.bins_spectre[:-1]+self.bins_spectre[1:])
        test = logical_and(t_middle > xmin, t_middle < xmax)
        ext_test = r_[test, [False]]
        ext_test = np.logical_or(ext_test, np.roll(ext_test, 1))
        tx, ty = self.bins_spectre[
            ext_test], self.spectre[test]
        self.bins_computation, self.data_computation = tx, ty
        level_y = log10(ty.mean()) if self.zoom_widget.getAxis('left').logMode \
            else ty.mean()
        pos = [(tx[0], level_y),  (tx[-1], level_y)]
        self.manualPolyaPriorPlot.setData(pos=np.array(pos), brush=QColor(Qt.cyan),  # , text="du text"
                                          pen=pg.mkPen(pg.mkColor(QColor(Qt.yellow)), width=1))  # , adj=adj, size=.2, symbol=symbols, pxMode=False, text=texts)

    def StopCompute(self):
        # -> STOP
        self.state_computation = STOP
        self.actionStartPause.setChecked(False)

    def StartPauseCompute(self):
        if self.state_computation == START:  # -> PAUSE
            self.state_computation = PAUSE

        elif self.state_computation == STOP:  # -> START + new computation
            self.set_data_computation()
            self.start_compute_thread()
            self.state_computation = START

        elif self.state_computation == PAUSE:  # -> START
            self.state_computation = START

        else:
            pass

    def getDataForPlotPics(self, p_energies, p_intensities, ref):
        assert len(p_energies) == len(p_intensities), 'pb getDataForPlotPics'
        nb_pics = len(p_energies)
        y0 = np.interp(
            p_energies, self.bins_computation[:-1], ref)

        trans = (lambda x: log10(x)) if self.zoom_widget.getAxis('left').logMode \
            else (lambda x: x)

        pos = hstack((array([p_energies, trans(p_intensities+y0)]),
                      array([p_energies, trans(y0)]))).T

        adj = c_[arange(nb_pics), arange(nb_pics)+nb_pics]
        symbols = np.repeat(["o", "o"], nb_pics).T
        size = np.repeat([6, 0], nb_pics).T
        symbolPen = pg.mkPen(pg.mkColor(QColor(Qt.red)))
        symbolBrush = QColor('#FFA500')
        return dict(pos=pos, adj=adj, symbol=symbols, symbolPen=symbolPen, size=size,
                    symbolBrush=symbolBrush, pen=pg.mkPen(pg.mkColor(QColor(Qt.red))))

    def progress_fn(self, n_iterations, dataDico):
        # self.bar.setValue(n)
        sumSpectrum = dataDico["N"]
        background = sumSpectrum*dataDico["fond"]
        self.PlotBackground.setData(
            self.bins_computation, background)
        self.PlotReconstruction.setData(
            self.bins_computation, sumSpectrum*dataDico["reconstruction"])
        self.PlotPics.setData(**self.getDataForPlotPics(dataDico["pics_energies"],
                                                        sumSpectrum *
                                                        dataDico["pics_weights"],
                                                        background))
        print(n_iterations, time.time())

    def GibbsSampler(self, progress_callback, paramGibbs):
        # -----------------------------------------
        # parameters
        data = array(self.data_computation, dtype=np.int64)
        numcans = len(data)
        offset_energy_canal = self.bins_computation[0]
        coef_energy_canal = (
            self.bins_computation[-1]-self.bins_computation[0])/numcans
        print("kevcan", coef_energy_canal)

        offset_variance_energy = 0.063555
        coef_variance_energy = 0.00037314

        numthreads = self.threadpool.maxThreadCount()
        print("numthreads", numthreads)
        n_per_call = 10
        tab_energ_bins = .5 * \
            (self.bins_computation[:-1]+self.bins_computation[1:])

        compton = np.empty_like(data, dtype=pl.float64)
        reconstruction = np.empty_like(data, dtype=pl.float64)
        # INITIALISATION
        MAX_NUMBER_OF_PICS = 1000
        pics_weights = zeros(MAX_NUMBER_OF_PICS, dtype=pl.float64)
        pics_energies = empty(MAX_NUMBER_OF_PICS, dtype=pl.float64)
        pics_variances = empty(MAX_NUMBER_OF_PICS, dtype=pl.float64)

        def initialisationComputation(p_weights, p_energies, p_variances, backgroung,
                                      _a_polya, _p_polya,
                                      prior_polya_in):
            e = tab_energ_bins[0]
            id_pic = 0
            while e < tab_energ_bins[-1]:
                variance = coef_variance_energy * e + offset_variance_energy
                p_energies[id_pic] = e
                p_variances[id_pic] = variance
                e += 5*sqrt(variance)
                p_weights[id_pic] = 1.
                id_pic += 1
            p_weights[:id_pic] /= sum(p_weights[:id_pic])
            backgroung[:] = lib_sinbad5.polya_parallel(data,
                                                       exp(_p_polya * (8 -
                                                                       np.floor(np.log2(numcans)))), _p_polya,
                                                       prior_polya_in, 1.)

            # pics/compton proportions
            prop_pics_background = .01
            backgroung *= 1-prop_pics_background
            p_weights[:id_pic] *= prop_pics_background

            print(f"Initialisation Computation: {id_pic} pics")
            return prop_pics_background, id_pic

        ####### COMPUTATION LOOP #####################
        initialisation_request = True
        while True:
            ni = 0
            #### read param ####
            # background
            h_polya = self.sliderPolyaH.value()
            m_polya = self.sliderPolyaM.value()
            h_polya /= 15.
            m_polya = exp((m_polya-np.log2(numcans))*h_polya)

            x_polya_prior, y_polya_prior = array(
                self.manualPolyaPriorPlot.pos).T

            alpha_dirichlet = 10**(.1*self.sliderAlphaDirichlet.value())
            trans = (lambda x: 10**x) if self.zoom_widget.getAxis('left').logMode \
                else (lambda x: x)

            y_polya_prior = trans(y_polya_prior)  # log scale
            ind_sort = np.argsort(x_polya_prior)
            prior_polya = np.interp(
                tab_energ_bins, x_polya_prior[ind_sort], y_polya_prior[ind_sort])
            # mean(prior_polya) must be 1.
            prior_polya *= numcans/sum(prior_polya)
            if initialisation_request:
                initialisation_request = False
                proportionPicsFond, nb_pics = initialisationComputation(
                    pics_weights, pics_energies, pics_variances, compton, m_polya, h_polya, prior_polya)
                # print(pics_weights[:nb_pics])
                # print(pics_energies[:nb_pics])
                # print(sqrt(pics_variances[:nb_pics])*2.35)

            # Computation for a half second
            # t0 = time.time()
            # while time.time()-t0 < .5:
            #     if self.state_computation == STOP:
            #         break
            proportionPicsFond, nb_pics, nb_pics_aff = lib_sinbad5.iterations(
                n_per_call, data, nb_pics, proportionPicsFond,
                compton, pics_weights, pics_energies, pics_variances, reconstruction,
                m_polya, h_polya, prior_polya, 1.,
                offset_energy_canal, coef_energy_canal, offset_variance_energy, coef_variance_energy,
                alpha_dirichlet)

            nb_pics = np.int64(nb_pics)
            nb_pics_aff = np.int64(nb_pics_aff)
            # print("nb_pics",nb_pics)
            # return "break 0"
            while self.state_computation == PAUSE:
                time.sleep(.2)

                # ni += n_per_call
            print('nb_pics', nb_pics, nb_pics_aff)
            if self.state_computation == STOP:
                break
            progress_callback.emit(ni, dict(N=sum(data),
                                            fond=np.copy(compton),
                                            reconstruction=np.copy(
                                                reconstruction),
                                            pics_weights=pics_weights[:nb_pics_aff],
                                            pics_variances=pics_variances[:nb_pics_aff],
                                            pics_energies=pics_energies[:nb_pics_aff]))

        return "Done."

    def print_output(self, s):
        print("final result:", s)

    def thread_complete(self):
        print("END COMPUTATION")

    def start_compute_thread(self):
        # Pass the function to execute
        # Any other args, kwargs are passed to the run function
        # paramGibbs = np.load("dicoFred.npy", allow_pickle=True,
        #                      encoding='latin1').tolist()
        worker = Worker(self.GibbsSampler, paramGibbs=None)

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
