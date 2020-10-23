# add licence for DAT online
from dendritic_arborization_tracer.minimal_DAT import threshold_neuron, detect_cell_body, watershed_segment_neuron, \
    skel_segment_neuronal_mask, prune_dendrites, find_neurons, detect_cell_bonds, get_cell_body_outline, Img
from dendritic_arborization_tracer.paint import Createpaintwidget
from PyQt5.QtWidgets import QToolBar, QListWidgetItem, QAbstractItemView, QSpinBox, QComboBox, QProgressBar, \
    QVBoxLayout, QLabel, QDoubleSpinBox
from PyQt5.QtCore import QSize, QItemSelectionModel, Qt
from PyQt5.QtGui import QPalette, QPixmap
from PyQt5.QtWidgets import QAction, QScrollArea, QStackedWidget
from PyQt5.QtWidgets import QWidget, QGridLayout, QPushButton, QToolButton, QListWidget, QFrame, QTabWidget
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMenu, QApplication
from timeit import default_timer as timer
from PyQt5 import QtWidgets, QtCore, QtGui
import numpy as np
import traceback
import os
from skimage.morphology import remove_small_objects
# logging
from dendritic_arborization_tracer.logger import TA_logger

logger = TA_logger()

QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)

__MAJOR__ = 0
__MINOR__ = 1
__MICRO__ = 0
__RELEASE__ = ''  # https://www.python.org/dev/peps/pep-0440/#public-version-identifiers --> alpha beta, ...
__VERSION__ = ''.join(
    [str(__MAJOR__), '.', str(__MINOR__), '.'.join([str(__MICRO__)]) if __MICRO__ != 0 else '', __RELEASE__])
__AUTHOR__ = 'Benoit Aigouy'
__EMAIL__ = 'baigouy@gmail.com'
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
            QtCore.QRect(centerPoint.x() - int(window_width / 2), centerPoint.y() - int(window_height / 2), window_width,
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

        # listen to tab changes
        self.tabs.currentChanged.connect(self._onTabChange)

        # Create first tab
        self.tab1.layout = QGridLayout()
        self.tab1.layout.setAlignment(Qt.AlignTop)
        self.tab1.layout.setHorizontalSpacing(3)
        self.tab1.layout.setVerticalSpacing(3)
        self.tab1.layout.setContentsMargins(0, 0, 0, 0)

        label1_tab1 = QLabel('Step 1:')
        self.tab1.layout.addWidget(label1_tab1, 0, 0)

        self.local_threshold = QPushButton("Local threshold")
        self.local_threshold.clicked.connect(self.run_threshold_neuron)
        self.tab1.layout.addWidget(self.local_threshold, 0, 1)
        self.global_threshold = QPushButton("Global threshold")
        self.global_threshold.clicked.connect(self.run_threshold_neuron)
        self.tab1.layout.addWidget(self.global_threshold, 0, 2)
        self.local_n_global_threshold = QPushButton("Local AND Global threshold")
        self.local_n_global_threshold.clicked.connect(self.run_threshold_neuron)
        self.tab1.layout.addWidget(self.local_n_global_threshold, 0, 3)

        self.extra_value_for_threshold = QSpinBox()
        self.extra_value_for_threshold.setSingleStep(1)
        self.extra_value_for_threshold.setRange(0, 1_000_000)
        self.extra_value_for_threshold.setValue(6)
        self.tab1.layout.addWidget(self.extra_value_for_threshold, 0, 4)

        self.threshold_method = QComboBox()
        self.threshold_method.addItem('Mean')
        self.threshold_method.addItem('Median')
        self.tab1.layout.addWidget(self.threshold_method, 0, 5)

        label2_tab1 = QLabel('Step 2 (optional):')
        self.tab1.layout.addWidget(label2_tab1, 1, 0)

        self.remove_pixel_blobs_smaller_or_equal = QPushButton("Remove pixel blobs smaller or equal to")
        self.remove_pixel_blobs_smaller_or_equal.clicked.connect(self.remove_blobs)
        self.tab1.layout.addWidget(self.remove_pixel_blobs_smaller_or_equal, 1, 1)

        self.remove_blobs_size = QSpinBox()
        self.remove_blobs_size.setSingleStep(1)
        self.remove_blobs_size.setRange(0, 1_000_000)
        self.remove_blobs_size.setValue(1)
        self.tab1.layout.addWidget(self.remove_blobs_size, 1, 2)


        label3_tab1 = QLabel('Step 3: Save')
        self.tab1.layout.addWidget(label3_tab1, 2, 0)

        self.tab1.setLayout(self.tab1.layout)

        self.tab2.layout = QGridLayout()
        self.tab2.layout.setAlignment(Qt.AlignTop)
        self.tab2.layout.setHorizontalSpacing(3)
        self.tab2.layout.setVerticalSpacing(3)
        self.tab2.layout.setContentsMargins(0, 0, 0, 0)

        label1_tab2 = QLabel('Step 1:')
        self.tab2.layout.addWidget(label1_tab2, 0, 0)

        self.detect_cell_body = QPushButton("Detect cell body")
        self.detect_cell_body.clicked.connect(self.detect_neuronal_cell_body)
        self.tab2.layout.addWidget(self.detect_cell_body, 0, 1)

        self.extraCutOff_cell_body = QSpinBox()
        self.extraCutOff_cell_body.setSingleStep(1)
        self.extraCutOff_cell_body.setRange(0, 1_000_000)
        self.extraCutOff_cell_body.setValue(5)
        self.tab2.layout.addWidget(self.extraCutOff_cell_body, 0, 2)

        erosion_label = QLabel('erosion rounds')
        self.tab2.layout.addWidget(erosion_label, 0, 3)

        self.nb_erosion_cellbody = QSpinBox()
        self.nb_erosion_cellbody.setSingleStep(1)
        self.nb_erosion_cellbody.setRange(0, 1_000_000)
        self.nb_erosion_cellbody.setValue(2)
        self.tab2.layout.addWidget(self.nb_erosion_cellbody, 0, 4)

        min_object_size_label = QLabel('minimum object size')
        self.tab2.layout.addWidget(min_object_size_label, 0, 5)

        self.min_obj_size_px = QSpinBox()
        self.min_obj_size_px.setSingleStep(1)
        self.min_obj_size_px.setRange(0, 1_000_000)
        self.min_obj_size_px.setValue(600)
        self.tab2.layout.addWidget(self.min_obj_size_px, 0, 6)

        fill_label = QLabel('fill up to')
        self.tab2.layout.addWidget(fill_label, 0, 7)

        self.fill_holes_up_to = QSpinBox()
        self.fill_holes_up_to.setSingleStep(1)
        self.fill_holes_up_to.setRange(0, 1_000_000)
        self.fill_holes_up_to.setValue(600)
        self.tab2.layout.addWidget(self.fill_holes_up_to, 0, 8)

        nb_dilation_cell_body_label = QLabel('nb dilation cell body')
        self.tab2.layout.addWidget(nb_dilation_cell_body_label, 0, 9)

        self.nb_dilation_cellbody = QSpinBox()
        self.nb_dilation_cellbody.setSingleStep(1)
        self.nb_dilation_cellbody.setRange(0, 1_000_000)
        self.nb_dilation_cellbody.setValue(2)
        self.tab2.layout.addWidget(self.nb_dilation_cellbody, 0, 10)

        label2_tab2 = QLabel('Step 2: Save')
        self.tab2.layout.addWidget(label2_tab2, 6, 0)

        self.tab2.setLayout(self.tab2.layout)

        self.tab3.layout = QGridLayout()
        self.tab3.layout.setAlignment(Qt.AlignTop)
        self.tab3.layout.setHorizontalSpacing(3)
        self.tab3.layout.setVerticalSpacing(3)
        self.tab3.layout.setContentsMargins(0, 0, 0, 0)

        label1_tab3 = QLabel('Step 1:')
        self.tab3.layout.addWidget(label1_tab3, 0, 0)

        self.wshed = QPushButton("Watershed")
        self.wshed.clicked.connect(self.watershed_segment_the_neuron)
        self.tab3.layout.addWidget(self.wshed, 0, 1)

        self.whsed_big_blur = QDoubleSpinBox()
        self.whsed_big_blur.setSingleStep(0.1)
        self.whsed_big_blur.setRange(0, 100)
        self.whsed_big_blur.setValue(2.1)
        self.tab3.layout.addWidget(self.whsed_big_blur, 0, 2)

        self.whsed_small_blur = QDoubleSpinBox()
        self.whsed_small_blur.setSingleStep(0.1)
        self.whsed_small_blur.setRange(0, 100)
        self.whsed_small_blur.setValue(1.4)
        self.tab3.layout.addWidget(self.whsed_small_blur, 0, 3)

        self.wshed_rm_small_cells = QSpinBox()
        self.wshed_rm_small_cells.setSingleStep(1)
        self.wshed_rm_small_cells.setRange(0, 1_000_000)
        self.wshed_rm_small_cells.setValue(10)
        self.tab3.layout.addWidget(self.wshed_rm_small_cells, 0, 4)

        self.jSpinner11 = QSpinBox()
        self.jSpinner11.setSingleStep(1)
        self.jSpinner11.setRange(0, 1_000_000)
        self.jSpinner11.setValue(10)
        self.tab3.layout.addWidget(self.jSpinner11, 0, 5)

        label1_bis_tab3 = QLabel('Alternative Step 1:')
        self.tab3.layout.addWidget(label1_bis_tab3, 1, 0)

        self.skel = QPushButton("Skeletonize")
        self.skel.clicked.connect(self.skeletonize_mask)
        self.tab3.layout.addWidget(self.skel, 1, 1)

        label2_tab3 = QLabel('Step 2:')
        self.tab3.layout.addWidget(label2_tab3, 2, 0)

        self.apply_cell_body = QPushButton("Apply cell body")
        self.apply_cell_body.clicked.connect(self.apply_cell_body_to_skeletonized_mask)
        self.tab3.layout.addWidget(self.apply_cell_body, 2, 1)

        label3_tab3 = QLabel('Step 3 (Optional):')
        self.tab3.layout.addWidget(label3_tab3, 3, 0)

        self.prune = QPushButton("Prune")
        self.prune.clicked.connect(self.prune_dendrites)
        self.tab3.layout.addWidget(self.prune, 3, 1)

        self.prune_length = QSpinBox()
        self.prune_length.setSingleStep(1)
        self.prune_length.setRange(0, 1_000_000)
        self.prune_length.setValue(3)
        self.tab3.layout.addWidget(self.prune_length, 3, 2)

        label4_tab3 = QLabel('Step 4 (Optional):')
        self.tab3.layout.addWidget(label4_tab3, 4, 0)

        self.find_neurons = QPushButton("Find neurons")
        self.find_neurons.clicked.connect(self.find_neurons_in_mask)
        self.tab3.layout.addWidget(self.find_neurons, 4, 1)

        self.find_neurons_min_size = QSpinBox()
        self.find_neurons_min_size.setSingleStep(1)
        self.find_neurons_min_size.setRange(0, 1_000_000)
        self.find_neurons_min_size.setValue(45)
        self.tab3.layout.addWidget(self.find_neurons_min_size, 4, 2)

        self.prune_unconnected_segments = QPushButton("Prune unconnected segments (run 'Find neurons' first)")
        self.prune_unconnected_segments.clicked.connect(self.prune_neuron_unconnected_segments)
        self.tab3.layout.addWidget(self.prune_unconnected_segments, 4, 3)

        label6_tab3 = QLabel('Step 5: Save')
        self.tab3.layout.addWidget(label6_tab3, 5, 0)

        label5_tab3 = QLabel('Step 6:')
        self.tab3.layout.addWidget(label5_tab3, 6, 0)

        self.create_n_save_bonds = QPushButton("Segment dendrites")
        self.create_n_save_bonds.clicked.connect(self.save_segmented_bonds)
        self.tab3.layout.addWidget(self.create_n_save_bonds, 6, 1)

        self.tab3.setLayout(self.tab3.layout)

        # Add tabs to widget
        table_widget_layout.addWidget(self.tabs)
        self.table_widget.setLayout(table_widget_layout)

        self.Stack = QStackedWidget(self)
        self.Stack.addWidget(self.scrollArea)

        # create a grid that will contain all the GUI interface
        self.grid = QGridLayout()
        self.grid.addWidget(self.Stack, 0, 0)
        self.grid.addWidget(self.list, 0, 1)
        # The first parameter of the rowStretch method is the row number, the second is the stretch factor. So you need two calls to rowStretch, like this: --> below the first row is occupying 80% and the second 20%
        self.grid.setRowStretch(0, 75)
        self.grid.setRowStretch(2, 25)

        # first col 75% second col 25% of total width
        self.grid.setColumnStretch(0, 75)
        self.grid.setColumnStretch(1, 25)

        # void QGridLayout::addLayout(QLayout * layout, int row, int column, int rowSpan, int columnSpan, Qt::Alignment alignment = 0)
        self.grid.addWidget(self.table_widget, 2, 0, 1, 2)  # spans over one row and 2 columns

        # BEGIN TOOLBAR
        # pen spin box and connect
        self.penSize = QSpinBox()
        self.penSize.setSingleStep(1)
        self.penSize.setRange(1, 256)
        self.penSize.setValue(3)
        self.penSize.valueChanged.connect(self.penSizechange)

        self.channels = QComboBox()
        self.channels.addItem("merge")
        self.channels.currentIndexChanged.connect(self.channelChange)

        tb_drawing_pane = QToolBar()

        save_button = QToolButton()
        save_button.setText("Save")
        save_button.clicked.connect(self.save_current_mask)
        tb_drawing_pane.addWidget(save_button)

        tb_drawing_pane.addWidget(QLabel("Channels"))
        tb_drawing_pane.addWidget(self.channels)

        # tb.addAction("Save")
        #
        tb_drawing_pane.addWidget(QLabel("Pen size"))
        tb_drawing_pane.addWidget(self.penSize)

        self.grid.addWidget(tb_drawing_pane, 1, 0)
        # END toolbar

        # toolbar for the list
        tb_list = QToolBar()

        del_button = QToolButton()
        del_button.setText("Delete selection from list")
        del_button.clicked.connect(self.delete_from_list)
        tb_list.addWidget(del_button)

        self.grid.addWidget(tb_list, 1, 1)


        # self.setCentralWidget(self.scrollArea)
        self.setCentralWidget(QFrame())
        self.centralWidget().setLayout(self.grid)

        # self.statusBar().showMessage('Ready')
        statusBar = self.statusBar()  # sets an empty status bar --> then can add messages in it
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

        # set drawing window fullscreen
        fullScreenShortcut = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_F), self)
        fullScreenShortcut.activated.connect(self.fullScreen)
        fullScreenShortcut.setContext(QtCore.Qt.ApplicationShortcut)  

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

        # connect enter keys to edit dendrites
        enterShortcut = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Return), self)
        enterShortcut.activated.connect(self.runSkel)
        enterShortcut.setContext(QtCore.Qt.ApplicationShortcut)  
        enter2Shortcut = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Enter), self)
        enter2Shortcut.activated.connect(self.runSkel)
        enter2Shortcut.setContext(QtCore.Qt.ApplicationShortcut)  

        #Qt.Key_Enter is the Enter located on the keypad:
        #Qt::Key_Return  0x01000004
        #Qt::Key_Enter   0x01000005  Typically located on the keypad.

        self.setAcceptDrops(True)  # KEEP IMPORTANT

    def __get_mask_img_from_overlay(self):
        if self.paint.imageDraw:
            channels_count = 4
            s = self.paint.imageDraw.bits().asstring(self.img.shape[0] * self.img.shape[1] * channels_count)
            arr = np.frombuffer(s, dtype=np.uint8).reshape((self.img.shape[0], self.img.shape[1], channels_count))
            return Img(arr[..., 2].copy(), dimensions='hw')
        else:
            return None

    def __get_output_folder(self):
        selected_items = self.list.selectedItems()
        if selected_items:
            filename = selected_items[0].toolTip()
            filename0_without_ext = os.path.splitext(filename)[0]
            return filename0_without_ext
        else:
            return None

    def delete_from_list(self):
        list_items = self.list.selectedItems()
        # empty list --> nothing to do
        if not list_items:
            return
        for item in list_items:
            self.list.takeItem(self.list.row(item))

    def save_current_mask(self):
        output_folder = self.__get_output_folder()
        if output_folder is None:
            logger.error('No image is selected --> nothing to save')
            return
        mask = self.__get_mask_img_from_overlay()
        if mask is None:
            logger.error('No mask/overlay detected --> nothing to save')
            return
        if self.tabs.currentIndex() == 0:
            print('saving', os.path.join(output_folder, 'mask.tif'))
            mask.save(os.path.join(output_folder, 'mask.tif'))
        elif self.tabs.currentIndex() == 1:
            print('saving', os.path.join(output_folder, 'cellBodyMask.tif'))
            mask.save(os.path.join(output_folder, 'cellBodyMask.tif'))
        else:
            print('saving', os.path.join(output_folder, 'handCorrection.tif'))
            mask.save(os.path.join(output_folder, 'handCorrection.tif'))

    def detect_neuronal_cell_body(self):
        try:
            # get image and detect cell body
            mask = detect_cell_body(self.img, fillHoles=self.fill_holes_up_to.value(),
                                    denoise=self.min_obj_size_px.value(), nbOfErosions=self.nb_erosion_cellbody.value(),
                                    nbOfDilatations=self.nb_dilation_cellbody.value(),
                                    extraCutOff=self.extraCutOff_cell_body.value(), channel=self.channels.currentText())
            if mask is not None:
                self.paint.imageDraw = Img(self.createRGBA(mask), dimensions='hwc').getQimage()
                self.paint.update()
            else:
                logger.error('Cell body could not be detected')
        except:
            traceback.print_exc()

    def __get_neuronal_mask(self, warn=True):
        output_folder = self.__get_output_folder()
        if output_folder is None:
            if warn:
                logger.error('No image selected --> nothing to do')
            return None
        if os.path.exists(os.path.join(output_folder, 'mask.tif')):
            # NB should I check the nb of channels --> no I don't want to handle externally created files and want people to rely fully on my stuff that has
            return Img(os.path.join(output_folder, 'mask.tif'))
        else:
            if warn:
                logger.error('Neuronal mask not found --> Please create one in the "Mask neuron" tab first')
        return None

    def __get_corrected_mask(self, warn=True):
        output_folder = self.__get_output_folder()
        if output_folder is None:
            if warn:
                logger.error('No image selected --> nothing to do')
            return None
        if os.path.exists(os.path.join(output_folder, 'handCorrection.tif')):
            return Img(os.path.join(output_folder, 'handCorrection.tif'))
        elif os.path.exists(os.path.join(output_folder, 'mask.tif')) and not warn:
            return Img(os.path.join(output_folder, 'mask.tif'))
        return None

    def __get_cellbody_mask(self, warn=True):
        output_folder = self.__get_output_folder()
        if output_folder is None:
            if warn:
                logger.error('No image selected --> nothing to do')
            return None
        if os.path.exists(os.path.join(output_folder, 'cellBodyMask.tif')):
            # NB should I check the nb of channels --> no I don't want to handle externally created files and want people to rely fully on my stuff that has
            return Img(os.path.join(output_folder, 'cellBodyMask.tif'))
        else:
            if warn:
                logger.error('Cell body mask not found --> Please create one in the "Mask cell body" tab first')
        return None

    # seems ok for now
    def watershed_segment_the_neuron(self):
        try:
            # get raw image and segment it using the watershed algorithm
            # make it load the neuronal mask
            neuronal_mask = self.__get_neuronal_mask()
            if neuronal_mask is None:
                return
            # TODO should I add autoskel or not
            mask = watershed_segment_neuron(self.img, neuronal_mask, fillSize=self.jSpinner11.value(), autoSkel=True,
                                            first_blur=self.whsed_big_blur.value(),
                                            second_blur=self.whsed_small_blur.value(),
                                            min_size=self.wshed_rm_small_cells.value(), channel=self.channels.currentText())
            if mask is not None:
                self.paint.imageDraw = Img(self.createRGBA(mask), dimensions='hwc').getQimage()
                self.paint.update()
            else:
                logger.error('Something went wrong, the neuron could not be segmented, sorry...')
        except:
            traceback.print_exc()

    def save_segmented_bonds(self):
        output_folder = self.__get_output_folder()
        if output_folder is None:
            logger.error('No image is selected --> nothing to save')
            return
        # get mask the find neurons
        mask = self.__get_mask_img_from_overlay()
        if mask is None:
            logger.error('No mask/overlay detected --> nothing to do')
            return
        mask = detect_cell_bonds(mask)

        if mask is None:
            logger.error('Could not find dendrites, are you sure a mask is overlayed over the neuron')
            return

        # code for conversion of 24 bits numpy array to an RGB one --> keep and store in Img at some point cause useful
        # convert 24 bits array to RGB
        RGB_mask = np.zeros(shape=(*mask.shape, 3), dtype=np.uint8)
        RGB_mask[..., 2] = mask & 255
        RGB_mask[..., 1] = (mask >> 8) & 255
        RGB_mask[..., 0] = (mask >> 16) & 255

        Img(RGB_mask, dimensions='hwc').save(os.path.join(output_folder, 'bonds.tif'))

    def prune_neuron_unconnected_segments(self):
        # get mask the find neurons
        mask = self.__get_mask_img_from_overlay()
        if mask is None:
            logger.error('No mask/overlay detected --> nothing to do')
            return
        mask = find_neurons(mask, neuron_minimum_size_threshold=self.find_neurons_min_size.value(),
                            return_unconnected=True)
        if mask is None:
            logger.error('Could not find neurons, are you sure a mask is overlayed over the neuron')
            return

        self.paint.imageDraw = Img(self.createRGBA(mask), dimensions='hwc').getQimage()
        self.paint.update()

    def apply_cell_body_to_skeletonized_mask(self):
        mask = self.__get_mask_img_from_overlay()
        if mask is None:
            logger.error('No mask/overlay detected --> nothing to do')
            return
        cell_body_mask = self.__get_cellbody_mask()
        if cell_body_mask is None:
            return
        cell_body_outline = get_cell_body_outline(cell_body_mask, mask)
        if cell_body_outline is None:
            logger.error('Error could not add cell body outline to the neuronal mask...')
            return
        self.paint.imageDraw = Img(self.createRGBA(cell_body_outline), dimensions='hwc').getQimage()
        self.paint.update()

    def find_neurons_in_mask(self):
        # get mask the find neurons
        mask = self.__get_mask_img_from_overlay()
        if mask is None:
            logger.error('No mask/overlay detected --> nothing to do')
            return
        mask_copy = mask.copy()
        mask = find_neurons(mask, neuron_minimum_size_threshold=self.find_neurons_min_size.value())
        if mask is None:
            logger.error('Could not find neurons, are you sure a mask is overlayed over the neuron')
            return

        # we set the red channel, the blue one, the alpha transparency (channel 4) and finally we only allow alpha channel in the two masks regions to keep the rest of the stuff
        final_overlay = np.zeros(shape=(*mask_copy.shape, 4), dtype=np.uint8)
        final_overlay[..., 0] = np.logical_xor(mask, mask_copy).astype(np.uint8) * 255  # blue channel
        final_overlay[mask == 0, 0] = 0
        final_overlay[..., 1] = final_overlay[..., 0]  # green channel # copy the channel to make the stuff appear cyan
        final_overlay[..., 2] = mask_copy  # red channel
        final_overlay[np.logical_or(mask, mask_copy) != 0, 3] = 255  # --> need set alpha transparency of the image

        self.paint.imageDraw = Img(final_overlay, dimensions='hwc').getQimage()
        self.paint.update()

    def prune_dendrites(self):

        prune_lgth = self.prune_length.value()
        if prune_lgth <= 0:
            logger.info('prune length is 0 --> nothing to do')
            return

        # get the mask from displayed image
        mask = self.__get_mask_img_from_overlay()
        if mask is None:
            logger.error('No mask/overlay detected --> nothing to do')
            return

        # see how to get the stuff ????
        mask = prune_dendrites(mask, prune_below=prune_lgth)

        if mask is None:
            logger.error('Could not prune dendrites, are you sure there is a mask ovrlayed on the neuron')
            return

        self.paint.imageDraw = Img(self.createRGBA(mask), dimensions='hwc').getQimage()
        self.paint.update()

    def skeletonize_mask(self):
        # get mask then skeletonize it then return it --> see exactly
        try:
            # get raw image and segment it using the skeletonize algorithm
            # make it load the neuronal mask
            neuronal_mask = self.__get_neuronal_mask()
            if neuronal_mask is None:
                return
            mask = skel_segment_neuronal_mask(neuronal_mask)
            if mask is not None:
                self.paint.imageDraw = Img(self.createRGBA(mask), dimensions='hwc').getQimage()
                self.paint.update()
            else:
                logger.error('Something went wrong, the neuron could not be sekeletonized, sorry...')
        except:
            traceback.print_exc()

    def _onTabChange(self):
        # if tab is changed --> do stuff
        # load files or warn...
        if self.tabs.currentIndex() == 0:
            mask = self.__get_neuronal_mask(warn=False)
            if mask is not None:
                self.paint.imageDraw = Img(self.createRGBA(mask), dimensions='hwc').getQimage()
                self.paint.update()
        elif self.tabs.currentIndex() == 1:
            mask = self.__get_cellbody_mask(warn=False)
            if mask is not None:
                self.paint.imageDraw = Img(self.createRGBA(mask), dimensions='hwc').getQimage()
                self.paint.update()
        elif self.tabs.currentIndex() == 2:
            mask = self.__get_corrected_mask(warn=False)
            if mask is not None:
                self.paint.imageDraw = Img(self.createRGBA(mask), dimensions='hwc').getQimage()
                self.paint.update()

    def run_threshold_neuron(self):
        try:
            local_or_global = 'global'
            if self.sender() == self.local_threshold:
                local_or_global = 'local'
            elif self.sender() == self.local_n_global_threshold:
                local_or_global = 'local+global'

            mask = threshold_neuron(self.img, mode=local_or_global, blur_method=self.threshold_method.currentText(),
                                    spin_value=self.extra_value_for_threshold.value(), channel=self.channels.currentText())
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
        # try update also the masks if they are available
        try:
            self._onTabChange()
        except:
            pass

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

    def nextFrame(self):
        idx = self.list.model().index(self.list.currentRow() + 1, 0)
        if idx.isValid():
            self.list.selectionModel().setCurrentIndex(idx, QItemSelectionModel.ClearAndSelect)  # SelectCurrent

    def remove_blobs(self):
        blob_size = self.remove_blobs_size.value()
        if blob_size <= 0:
            logger.info('blob size is 0 --> nothing to do')
            return

        # get the mask from displayed image
        mask = self.__get_mask_img_from_overlay()
        if mask is None:
            logger.error('No mask/overlay detected --> nothing to save')
            return

        mask = remove_small_objects(mask.astype(np.bool), min_size=blob_size, connectivity=2, in_place=False)
        # then place back pixels in the mask
        # now set the mask back
        # plt.imshow(mask)
        # plt.show()

        self.paint.imageDraw = Img(self.createRGBA(mask), dimensions='hwc').getQimage()
        self.paint.update()

    def runSkel(self):
        # only allow that for tab 3
        if self.tabs.currentIndex()==2:
            mask = self.__get_mask_img_from_overlay()
            if mask is None:
                logger.error('No mask/overlay detected --> nothing to do')
                return
            # just skeletonize the image
            mask = skel_segment_neuronal_mask(mask, fill_holes=0)  # should I put it to 0 or other things ???
            if mask is None:
                logger.error('Could not skeletonize user edited mask...')
                return

            self.paint.imageDraw = Img(self.createRGBA(mask), dimensions='hwc').getQimage()
            self.paint.update()

    def createRGBA(self, handCorrection):
        # use pen color to display the mask
        # in fact I need to put the real color
        RGBA = np.zeros((handCorrection.shape[0], handCorrection.shape[1], 4), dtype=np.uint8)
        red = self.paint.drawColor.red()
        green = self.paint.drawColor.green()
        blue = self.paint.drawColor.blue()

        # bug somewhere --> fix it some day --> due to bgra instead of RGBA
        RGBA[handCorrection != 0, 0] = blue  # b
        RGBA[handCorrection != 0, 1] = green  # g
        RGBA[handCorrection != 0, 2] = red  # r
        RGBA[..., 3] = 255  # alpha --> indeed alpha
        RGBA[handCorrection == 0, 3] = 0  # very complex fix some day

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
        # self.paint.vdp.scale = self.scale

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
