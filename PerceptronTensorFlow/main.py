import sys
from PyQt5 import uic, QtWidgets, QtGui
from PyQt5.QtWidgets import QMessageBox
import perceptron as neurona
import random as rand
qtCreatorFile = './view/mainView.ui' #Nombre del archivo
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)


class Main(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.setWindowTitle('.:: PERCEPTRÓN TENSORFLOW ::. 193258 - 193213')
        self.setFixedSize(516, 400)

        self.tasaAprendizaje.setValidator(QtGui.QDoubleValidator())
        self.epocas.setValidator(QtGui.QIntValidator())

        self.btnEjecutar.clicked.connect(self.iniciar)
        self.btnSalir.clicked.connect(self.salir)
    
    def iniciar(self):

        tasaAprendizaje, epocas = self.getNK()

        if not tasaAprendizaje and not epocas:
            tasaAprendizaje = float(round(rand.uniform(1, 0), 3))
            epocas = rand.randint(1, 1000)
            neurona.entrenamientoNeurona(epocas, tasaAprendizaje)

        else:

            if (not tasaAprendizaje and epocas) or (not epocas and tasaAprendizaje):
                QMessageBox.warning(None, 'Campo Vacio', 'No deje un campo vacio!')
            else:
                tasaAprendizaje = float(self.tasaAprendizaje.text())
                epocas = int(self.epocas.text())
                neurona.entrenamientoNeurona(epocas, tasaAprendizaje)

    def getNK(self):
        tasaAprendizaje = self.tasaAprendizaje.text()
        epocas = self.epocas.text()
        return tasaAprendizaje, epocas

    def salir(self):
        salir = QMessageBox.question(
            self, 'Salir', '¿Esta seguro/a de salir?', QMessageBox.Yes, QMessageBox.No)
        if salir == QMessageBox.Yes:
            self.close()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = Main()
    window.show()
    sys.exit(app.exec_())