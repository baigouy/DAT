import os
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QToolBar, QListWidgetItem, QAbstractItemView, QSpinBox, QComboBox, QProgressBar, \
    QVBoxLayout, QLabel, QDoubleSpinBox
from PyQt5.QtCore import QSize, QItemSelectionModel
from PyQt5.QtGui import QPalette, QPixmap
from PyQt5.QtWidgets import QAction, QScrollArea, QStackedWidget
from PyQt5.QtWidgets import QWidget, QGridLayout, QPushButton, QToolButton, QListWidget, QFrame, QTabWidget
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMenu, QApplication
from epyseg.draw.widgets.paint import Createpaintwidget
from deprecated_demos.ta.wshed import Wshed
from timeit import default_timer as timer
from PyQt5 import QtWidgets, QtCore, QtGui
from epyseg.img import Img
import numpy as np
import traceback
from personal.DAT.minimal_DAT import threshold_neuron
# logging
from epyseg.tools.logger import TA_logger
logger = TA_logger()

QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)

__MAJOR__ = 0
__MINOR__ = 1
__MICRO__ = 0
__RELEASE__='b' #https://www.python.org/dev/peps/pep-0440/#public-version-identifiers --> alpha beta, ...
__VERSION__ = ''.join([str(__MAJOR__), '.', str(__MINOR__), '.'.join([str(__MICRO__)]) if __MICRO__ != 0 else '', __RELEASE__])
__AUTHOR__ = 'Benoit Aigouy'
__NAME__ = 'Dendritic Arborization Tracer'

class DAT_GUI(QtWidgets.QMainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()

    def initUI(self):
        screen = QApplication.desktop().screenNumber(QApplication.desktop().cursor().pos())
        centerPoint = QApplication.desktop().screenGeometry(screen).center()

        # should fit in 1024x768 (old computer screens)
        window_width = 900
        window_height = 700
        self.setGeometry(
            QtCore.QRect(centerPoint.x() - window_width / 2, centerPoint.y() - window_height / 2, window_width,
                         window_height))  # should I rather center on the screen

        # zoom parameters
        self.scale = 1.0
        self.min_scaling_factor = 0.1
        self.max_scaling_factor = 20
        self.zoom_increment = 0.05

        self.setWindowTitle(__NAME__ + ' v' + str(__VERSION__))

        self.paint = Createpaintwidget()

        # initiate 2D image for 2D display
        self.img = None

        self.list = QListWidget(self)  # a list that contains files to read or play with
        self.list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list.selectionModel().selectionChanged.connect(self.selectionChanged)  # connect it to sel change

        self.scrollArea = QScrollArea()
        self.scrollArea.setBackgroundRole(QPalette.Dark)
        self.scrollArea.setWidget(self.paint)
        self.paint.scrollArea = self.scrollArea

        self.table_widget = QWidget()
        table_widget_layout = QVBoxLayout()

        # Initialize tab screen
        self.tabs = QTabWidget(self)
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()

        # Add tabs
        self.tabs.addTab(self.tab1, "Mask neuron")
        self.tabs.addTab(self.tab2, "Mask cell body")
        self.tabs.addTab(self.tab3, "Segment dendrites")

        # Create first tab
        # TODO put grid layout
        #
        self.tab1.layout = QVBoxLayout()
        self.local_threshold = QPushButton("Local threshold")
        self.local_threshold.clicked.connect(self.run_threshold_neuron)
        self.tab1.layout.addWidget(self.local_threshold)
        self.global_threshold = QPushButton("Global threshold")
        self.global_threshold.clicked.connect(self.run_threshold_neuron)
        self.tab1.layout.addWidget(self.global_threshold)
        self.local_n_global_threshold = QPushButton("Local & Global threshold")
        self.local_n_global_threshold.clicked.connect(self.run_threshold_neuron)
        self.tab1.layout.addWidget(self.local_n_global_threshold)
        #
        self.extra_value_for_threshold = QSpinBox()
        self.extra_value_for_threshold.setSingleStep(1)
        self.extra_value_for_threshold.setRange(0, 1_000_000)  
        self.extra_value_for_threshold.setValue(6)
        self.tab1.layout.addWidget(self.extra_value_for_threshold)

        self.threshold_method = QComboBox()
        self.threshold_method.addItem('Mean')
        self.threshold_method.addItem('Median')
        self.tab1.layout.addWidget(self.threshold_method)

        self.pushButton4 = QPushButton("Remove pixel blobs smaller than ")
        # self.pushButton1.clicked.connect(self.run_watershed)
        self.tab1.layout.addWidget(self.pushButton4)

        self.remove_blobs_smaller_than = QSpinBox()
        self.remove_blobs_smaller_than.setSingleStep(1)
        self.remove_blobs_smaller_than.setRange(0, 1_000_000)  
        self.remove_blobs_smaller_than.setValue(1)
        self.tab1.layout.addWidget(self.remove_blobs_smaller_than)

        self.tab1.setLayout(self.tab1.layout)

        self.tab2.layout = QVBoxLayout()

        self.detect_cell_body = QPushButton("Detect cell body")
        # self.pushButton1.clicked.connect(self.run_watershed)
        self.tab2.layout.addWidget(self.detect_cell_body)

        self.extraCutOff_cell_body = QSpinBox()
        self.extraCutOff_cell_body.setSingleStep(1)
        self.extraCutOff_cell_body.setRange(0, 1_000_000)  
        self.extraCutOff_cell_body.setValue(5)
        self.tab2.layout.addWidget(self.extraCutOff_cell_body)

        self.nb_erosion_cellbody = QSpinBox()
        self.nb_erosion_cellbody.setSingleStep(1)
        self.nb_erosion_cellbody.setRange(0, 1_000_000)  
        self.nb_erosion_cellbody.setValue(2)
        self.tab2.layout.addWidget(self.nb_erosion_cellbody)


        self.prev_cell_body = QPushButton("Preview cell body")
        # self.pushButton1.clicked.connect(self.run_watershed)
        self.tab2.layout.addWidget(self.prev_cell_body)


        self.apply_cell_body = QPushButton("Apply cell body")
        # self.pushButton1.clicked.connect(self.run_watershed)
        self.tab2.layout.addWidget(self.apply_cell_body)


        self.min_obj_size_px = QSpinBox()
        self.min_obj_size_px.setSingleStep(1)
        self.min_obj_size_px.setRange(0, 1_000_000)  
        self.min_obj_size_px.setValue(600)
        self.tab2.layout.addWidget(self.min_obj_size_px)

        self.fill_holes_up_to = QSpinBox()
        self.fill_holes_up_to.setSingleStep(1)
        self.fill_holes_up_to.setRange(0, 1_000_000)  
        self.fill_holes_up_to.setValue(600)
        self.tab2.layout.addWidget(self.fill_holes_up_to)

        self.nb_dilation_cellbody = QSpinBox()
        self.nb_dilation_cellbody.setSingleStep(1)
        self.nb_dilation_cellbody.setRange(0, 1_000_000)  
        self.nb_dilation_cellbody.setValue(2)
        self.tab2.layout.addWidget(self.nb_dilation_cellbody)

        self.tab2.setLayout(self.tab2.layout)

        self.tab3.layout = QVBoxLayout()

        self.wshed = QPushButton("Watershed")
        # self.pushButton1.clicked.connect(self.run_watershed)
        self.tab3.layout.addWidget(self.wshed)

        self.whsed_big_blur = QDoubleSpinBox()
        self.whsed_big_blur.setSingleStep(0.1)
        self.whsed_big_blur.setRange(0, 100)  
        self.whsed_big_blur.setValue(2.1)
        self.tab3.layout.addWidget(self.whsed_big_blur)

        self.whsed_small_blur = QDoubleSpinBox()
        self.whsed_small_blur.setSingleStep(0.1)
        self.whsed_small_blur.setRange(0, 100)  
        self.whsed_small_blur.setValue(1.4)
        self.tab3.layout.addWidget(self.whsed_small_blur)

        self.wshed_rm_small_cells = QSpinBox()
        self.wshed_rm_small_cells.setSingleStep(1)
        self.wshed_rm_small_cells.setRange(0, 1_000_000)  
        self.wshed_rm_small_cells.setValue(10)
        self.tab2.layout.addWidget(self.wshed_rm_small_cells)


        self.skel = QPushButton("Skeletonize")
        # self.pushButton1.clicked.connect(self.run_watershed)
        self.tab3.layout.addWidget(self.skel)

        self.prune = QPushButton("Prune")
        # self.pushButton1.clicked.connect(self.run_watershed)
        self.tab3.layout.addWidget(self.prune)

        self.prune_length = QSpinBox()
        self.prune_length.setSingleStep(1)
        self.prune_length.setRange(0, 1_000_000)  
        self.prune_length.setValue(3)
        self.tab2.layout.addWidget(self.prune_length)

        self.find_neurons = QPushButton("Find neurons")
        # self.pushButton1.clicked.connect(self.run_watershed)
        self.tab3.layout.addWidget(self.find_neurons)

        self.find_neurons_min_size = QSpinBox()
        self.find_neurons_min_size.setSingleStep(1)
        self.find_neurons_min_size.setRange(0, 1_000_000)  
        self.find_neurons_min_size.setValue(45)
        self.tab2.layout.addWidget(self.find_neurons_min_size)


        self.prune_unconnected_segments = QPushButton("Prune unconnected segments (run 'Find neurons' first)")
        # self.pushButton1.clicked.connect(self.run_watershed)
        self.tab3.layout.addWidget(self.prune)


        self.create_n_save_bonds = QPushButton("Create and save bonds")
        # self.pushButton1.clicked.connect(self.run_watershed)
        self.tab3.layout.addWidget(self.create_n_save_bonds)

        # TODO connect all now and I'll be done...
        # self.tab3.layout.addWidget(QPushButton("test button"))
        self.tab3.setLayout(self.tab3.layout)



        # Add tabs to widget
        table_widget_layout.addWidget(self.tabs)
        self.table_widget.setLayout(table_widget_layout)

        self.Stack = QStackedWidget(self)
        self.Stack.addWidget(self.scrollArea)

        self.grid = QGridLayout()
        self.grid.addWidget(self.Stack, 0, 0) 
        self.grid.addWidget(self.list, 0, 1)
        self.grid.setRowStretch(0, 75)
        self.grid.setRowStretch(2, 25)

        self.grid.setColumnStretch(0, 75)
        self.grid.setColumnStretch(1, 25)

        self.grid.addWidget(self.table_widget, 2, 0, 1, 2)

        self.penSize = QSpinBox()
        self.penSize.setSingleStep(1)
        self.penSize.setRange(1, 256)
        self.penSize.setValue(3)
        self.penSize.valueChanged.connect(self.penSizechange)

        self.channels = QComboBox()
        self.channels.addItem("merge")
        self.channels.currentIndexChanged.connect(self.channelChange)

        tb = QToolBar()

        save_button = QToolButton()
        save_button.setText("Save")
        save_button.clicked.connect(self.save_current_mask)
        tb.addWidget(save_button)

        tb.addWidget(QLabel("Channels"))
        tb.addWidget(self.channels)

        # tb.addAction("Save")
        #
        tb.addWidget(QLabel("Pen size"))
        tb.addWidget(self.penSize)

        self.grid.addWidget(tb, 1, 0, 1, 2)

        self.setCentralWidget(QFrame())
        self.centralWidget().setLayout(self.grid)

        statusBar = self.statusBar() 
        self.paint.statusBar = statusBar

        # add progress bar to status bar
        self.progress = QProgressBar(self)
        self.progress.setGeometry(200, 80, 250, 20)
        statusBar.addWidget(self.progress)

        # Set up menu bar
        self.mainMenu = self.menuBar()

        self.zoomInAct = QAction("Zoom &In (25%)", self,  # shortcut="Ctrl++",
                                 enabled=True, triggered=self.zoomIn)
        self.zoomOutAct = QAction("Zoom &Out (25%)", self,  # shortcut="Ctrl+-",
                                  enabled=True, triggered=self.zoomOut)
        self.normalSizeAct = QAction("&Normal Size", self,  # shortcut="Ctrl+S",
                                     enabled=True, triggered=self.defaultSize)

        self.viewMenu = QMenu("&View", self)
        self.viewMenu.addAction(self.zoomInAct)
        self.viewMenu.addAction(self.zoomOutAct)
        self.viewMenu.addAction(self.normalSizeAct)

        self.menuBar().addMenu(self.viewMenu)

        self.setMenuBar(self.mainMenu)

        # Setup hotkeys for whole system
        # Delete selected vectorial objects
        # deleteShortcut = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Delete), self)
        # deleteShortcut.activated.connect(self.down)
        # deleteShortcut.setContext(QtCore.Qt.ApplicationShortcut)

        # set drawing window fullscreen
        fullScreenShortcut = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_F), self)
        fullScreenShortcut.activated.connect(self.fullScreen)
        fullScreenShortcut.setContext(QtCore.Qt.ApplicationShortcut)  

        # exit from full screen TODO add quit the app too ??
        escapeShortcut = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Escape), self)
        escapeShortcut.activated.connect(self.escape)
        escapeShortcut.setContext(QtCore.Qt.ApplicationShortcut)  

        # Show/Hide the mask
        escapeShortcut = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_H), self)
        escapeShortcut.activated.connect(self.showHideMask)
        escapeShortcut.setContext(QtCore.Qt.ApplicationShortcut)  

        zoomPlus = QtWidgets.QShortcut("Ctrl+Shift+=", self)
        zoomPlus.activated.connect(self.zoomIn)
        zoomPlus.setContext(QtCore.Qt.ApplicationShortcut)  

        zoomPlus2 = QtWidgets.QShortcut("Ctrl++", self)
        zoomPlus2.activated.connect(self.zoomIn)
        zoomPlus2.setContext(QtCore.Qt.ApplicationShortcut)  

        zoomMinus = QtWidgets.QShortcut("Ctrl+Shift+-", self)
        zoomMinus.activated.connect(self.zoomOut)
        zoomMinus.setContext(QtCore.Qt.ApplicationShortcut)  

        zoomMinus2 = QtWidgets.QShortcut("Ctrl+-", self)
        zoomMinus2.activated.connect(self.zoomOut)
        zoomMinus2.setContext(QtCore.Qt.ApplicationShortcut)  

        spaceShortcut = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Space), self)
        spaceShortcut.activated.connect(self.nextFrame)
        spaceShortcut.setContext(QtCore.Qt.ApplicationShortcut)  

        backspaceShortcut = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Backspace), self)
        backspaceShortcut.activated.connect(self.prevFrame)
        backspaceShortcut.setContext(QtCore.Qt.ApplicationShortcut)  

        enterShortcut = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Return), self)
        enterShortcut.activated.connect(self.runWshed)
        enterShortcut.setContext(QtCore.Qt.ApplicationShortcut)  

        self.setAcceptDrops(True)  # KEEP IMPORTANT

    def save_current_mask(self):
        if  self.tabs.currentIndex() == 0:
            if self.paint.imageDraw:
                channels_count = 4
                s = self.paint.imageDraw.bits().asstring(self.img.shape[0] * self.img.shape[1] * channels_count)
                arr = np.frombuffer(s, dtype=np.uint8).reshape((self.img.shape[0], self.img.shape[1], channels_count))

                selected_items = self.list.selectedItems()
                if selected_items:
                    filename = selected_items[0].toolTip()
                    filename0_without_ext = os.path.splitext(filename)[0]
                    print('saving', os.path.join(filename0_without_ext, 'mask.tif'))
                    Img(arr[...,2], dimensions='hw').save(os.path.join(filename0_without_ext, 'mask.tif'))
        elif self.tabs.currentIndex() == 1:
            if self.paint.imageDraw:
                channels_count = 4
                s = self.paint.imageDraw.bits().asstring(self.img.shape[0] * self.img.shape[1] * channels_count)
                arr = np.frombuffer(s, dtype=np.uint8).reshape((self.img.shape[0], self.img.shape[1], channels_count))

                selected_items = self.list.selectedItems()
                if selected_items:
                    filename = selected_items[0].toolTip()
                    filename0_without_ext = os.path.splitext(filename)[0]
                    print('saving', os.path.join(filename0_without_ext, 'cellBodyMask.tif'))
                    Img(arr[...,2], dimensions='hw').save(os.path.join(filename0_without_ext, 'cellBodyMask.tif'))
        else:
            if self.paint.imageDraw:
                channels_count = 4
                s = self.paint.imageDraw.bits().asstring(self.img.shape[0] * self.img.shape[1] * channels_count)
                arr = np.frombuffer(s, dtype=np.uint8).reshape((self.img.shape[0], self.img.shape[1], channels_count))

                selected_items = self.list.selectedItems()
                if selected_items:
                    filename = selected_items[0].toolTip()
                    filename0_without_ext = os.path.splitext(filename)[0]
                    print('saving', os.path.join(filename0_without_ext, 'handCorrection.tif'))
                    Img(arr[..., 2], dimensions='hw').save(os.path.join(filename0_without_ext, 'handCorrection.tif'))



    def run_threshold_neuron(self):
        try:
            local_or_global = 'global'
            if self.sender() == self.local_threshold:
                local_or_global = 'local'
            elif self.sender() == self.local_n_global_threshold:
                local_or_global = 'local+global'

            mask = threshold_neuron(self.img, mode=local_or_global, blur_method=self.threshold_method.currentText(), spin_value=self.extra_value_for_threshold.value())
            if mask is not None:
                self.paint.imageDraw = Img(self.createRGBA(mask), dimensions='hwc').getQimage()
                self.paint.update()
        except:
            traceback.print_exc()


    def channelChange(self, i):
        if self.Stack.currentIndex() == 0:
            if i == 0:
                self.paint.setImage(self.img)
            else:
                channel_img = self.img.imCopy(c=i - 1)
                self.paint.setImage(channel_img)
            self.paint.update()


    def penSizechange(self):
        self.paint.brushSize = self.penSize.value()

    def selectionChanged(self):
        self.paint.maskVisible = True
        selected_items = self.list.selectedItems()
        if selected_items:
            start = timer()
            if self.img is not None:
                # make sure we don't load the image twice
                if selected_items[0].toolTip() != self.img.metadata['path']:
                    self.img = Img(selected_items[0].toolTip())
                    logger.debug("took " + str(timer() - start) + " secs to load image")
                else:
                    logger.debug("image already loaded --> ignoring")
            else:
                self.img = Img(selected_items[0].toolTip())
                logger.debug("took " + str(timer() - start) + " secs to load image")

        if self.img is not None:
            selection = self.channels.currentIndex()
            self.channels.disconnect()
            self.channels.clear()
            comboData = ['merge']
            if self.img.has_c():
                for i in range(self.img.get_dimension('c')):
                    comboData.append(str(i))
            logger.debug('channels found ' + str(comboData))
            self.channels.addItems(comboData)
            if selection != -1 and selection < self.channels.count():
                self.channels.setCurrentIndex(selection)
            else:
                self.channels.setCurrentIndex(0)
            self.channels.currentIndexChanged.connect(self.channelChange)

        if selected_items:
            self.statusBar().showMessage('Loading ' + selected_items[0].toolTip())
            selection = self.channels.currentIndex()
            if selection == 0:
                self.paint.setImage(self.img)
            else:
                self.paint.setImage(self.img.imCopy(c=selection - 1))
            self.scaleImage(0)
            self.update()
            self.paint.update()

            if self.list.currentItem() and self.list.currentItem().icon().isNull():
                logger.debug('Updating icon')
                icon = QIcon(QPixmap.fromImage(self.paint.image))
                pixmap = icon.pixmap(24, 24)
                icon = QIcon(pixmap)
                self.list.currentItem().setIcon(icon)
        else:
            logger.debug("Empty selection")
            self.paint.image = None
            self.scaleImage(0)
            self.update()
            self.paint.update()
            self.img = None

    def clearlayout(self, layout):
        for i in reversed(range(layout.count())):
            layout.itemAt(i).widget().setParent(None)

    def showHideMask(self):
        self.paint.maskVisible = not self.paint.maskVisible
        self.paint.update()

    def escape(self):
        if self.Stack.isFullScreen():
            self.fullScreen()

    def fullScreen(self):
        if not self.Stack.isFullScreen():
            self.Stack.setWindowFlags(
                QtCore.Qt.Window |
                QtCore.Qt.CustomizeWindowHint |
                # QtCore.Qt.WindowTitleHint |
                # QtCore.Qt.WindowCloseButtonHint |
                QtCore.Qt.WindowStaysOnTopHint
            )
            self.Stack.showFullScreen()
        else:
            self.Stack.setWindowFlags(QtCore.Qt.Widget)
            self.grid.addWidget(self.Stack, 0, 0)
            # dirty hack to make it repaint properly --> obviously not all lines below are required but some are --> need test, the last line is key though
            self.grid.update()
            self.Stack.update()
            self.Stack.show()
            self.centralWidget().setLayout(self.grid)
            self.centralWidget().update()
            self.update()
            self.show()
            self.repaint()
            self.Stack.update()
            self.Stack.repaint()
            self.centralWidget().repaint()

    def down(self):
        if self.paint.vdp.active:
            self.paint.vdp.remove_selection()
            self.paint.update()

    def nextFrame(self):
        idx = self.list.model().index(self.list.currentRow() + 1, 0)
        if idx.isValid():
            self.list.selectionModel().setCurrentIndex(idx, QItemSelectionModel.ClearAndSelect)  # SelectCurrent

    def runWshed(self):
        if self.paint.imageDraw:
            TODO

            # handCorrection = Wshed.run(handCorrection, seeds='mask') #TODO urgent add min size

            # need an RGBA here
            # print(self.paint.imageDraw)
            self.paint.imageDraw = Img(self.createRGBA(handCorrection), dimensions='hwc').getQimage() # marche pas car besoin d'une ARGB
            # print(self.paint.imageDraw)
            self.paint.update()

    def createRGBA(self, handCorrection):
        # use pen color to display the mask
        # in fact I need to put the real color
        RGBA = np.zeros((handCorrection.shape[0], handCorrection.shape[1], 4), dtype=np.uint8)
        red = self.paint.drawColor.red()
        green = self.paint.drawColor.green()
        blue = self.paint.drawColor.blue()

        # bug somewhere --> fix it some day --> due to bgra instead of RGBA
        RGBA[handCorrection != 0, 0] = blue # b
        RGBA[handCorrection != 0, 1] = green # g
        RGBA[handCorrection != 0, 2] = red # r
        RGBA[..., 3] = 255 # alpha --> indeed alpha
        RGBA[handCorrection == 0, 3] = 0 # very complex fix some day

        return RGBA

    def prevFrame(self):
        idx = self.list.model().index(self.list.currentRow() - 1, 0)
        if idx.isValid():
            self.list.selectionModel().setCurrentIndex(idx, QItemSelectionModel.ClearAndSelect)


    def zoomIn(self):
        self.statusBar().showMessage('Zooming in',
                                     msecs=200)
        if self.Stack.currentIndex() == 0:
            self.scaleImage(self.zoom_increment)

    def zoomOut(self):
        self.statusBar().showMessage('Zooming out', msecs=200)
        if self.Stack.currentIndex() == 0:
            self.scaleImage(-self.zoom_increment)

    def defaultSize(self):
        self.paint.adjustSize()
        self.scale = 1.0
        self.scaleImage(0)

    def scaleImage(self, factor):
        self.scale += factor
        if self.paint.image is not None:
            self.paint.resize(self.scale * self.paint.image.size())
        else:
            # no image set size to 0, 0 --> scroll pane will auto adjust
            self.paint.resize(QSize(0, 0))
            self.scale -= factor  # reset zoom

        self.paint.scale = self.scale
        self.paint.vdp.scale = self.scale

        self.zoomInAct.setEnabled(self.scale < self.max_scaling_factor)
        self.zoomOutAct.setEnabled(self.scale > self.min_scaling_factor)

    # allow DND
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls:
            event.setDropAction(QtCore.Qt.CopyAction)
            event.accept()
        else:
            event.ignore()

    # handle DND on drop
    def dropEvent(self, event):
        if event.mimeData().hasUrls:
            event.setDropAction(QtCore.Qt.CopyAction)
            event.accept()
            urls = []
            for url in event.mimeData().urls():
                urls.append(url.toLocalFile())

            for url in urls:
                import os
                item = QListWidgetItem(os.path.basename(url), self.list)
                item.setToolTip(url)
                self.list.addItem(item)
        else:
            event.ignore()


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    w = DAT_GUI()
    w.show()
    sys.exit(app.exec_())
