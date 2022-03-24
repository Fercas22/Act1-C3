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

        self.tazaAprendizaje.setValidator(QtGui.QDoubleValidator())
        self.epocas.setValidator(QtGui.QIntValidator())

        self.btnEjecutar.clicked.connect(self.iniciar)
        self.btnSalir.clicked.connect(self.salir)
    
    def iniciar(self):

        tazaAprendizaje, epocas = self.getNK()

        if not tazaAprendizaje and not epocas:
            tazaAprendizaje = float(round(rand.uniform(1, 0), 3))
            epocas = rand.randint(1, 1000)
            neurona.entrenamientoNeurona(epocas, tazaAprendizaje)

        else:

            if (not tazaAprendizaje and epocas) or (not epocas and tazaAprendizaje):
                QMessageBox.warning(None, 'Campo Vacio', 'No deje un campo vacio!')
            else:
                tazaAprendizaje = float(self.tazaAprendizaje.text())
                epocas = int(self.epocas.text())
                neurona.entrenamientoNeurona(epocas, tazaAprendizaje)

    def getNK(self):
        tazaAprendizaje = self.tazaAprendizaje.text()
        epocas = self.epocas.text()
        return tazaAprendizaje, epocas

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