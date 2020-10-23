from dendritic_arborization_tracer.minimal_DAT_GUI import DAT_GUI
from PyQt5.QtWidgets import QApplication
import sys

if __name__ == '__main__':
    # run the DAT GUI
    # code can be run from the command line using 'python -m dendritic_arborization_tracer'
    # dendritic-arborization-tracer --> à tester en fait
    # or that dendritetracer to avoid pbs -->
    app = QApplication(sys.argv)
    w = DAT_GUI()
    w.show()
    sys.exit(app.exec_())

