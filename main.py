import csv
import numpy as np
import math
import matplotlib.pyplot as plt

def main():
    # Lectura del CSV y extraccion de datos
    with open('datos.csv', newline='') as File:  
        reader = csv.reader(File)
        auxLit = []
        listaDatos = []
        for row in reader:
            for datos in row:
                auxLit.append(datos)
            listaDatos.append(auxLit)
            auxLit = []

    dataX = []
    dataY = []
    for bX in range(1, len(listaDatos)):
        dataX.append([1, float(listaDatos[bX][0])])
        dataY.append(float(listaDatos[bX][1]))
   
    X = np.array(dataX)
    print(dataY)

    X = X.transpose()
    Y = np.array(dataY)

    def entranamiento_Neurona(n, wk):
        k = 0
        errores = []
        generaciones = []
        while(True):
            k += 1
            uk = np.dot(wk, X)
            yck = np.array([0 if uk[0][i] < 0 else 1 for i in range(len(uk[0]))])
            ek = Y - yck

            temp = np.dot(X, ek) * n
            wt = wk + temp
            cont = 0

            for i in range(len(ek)):
                cont += ek[i]**2

            wk = wt
            errores.append((math.sqrt(cont)))
            generaciones.append(k)
        

            if np.all(yck == Y):
                return errores, generaciones, list(wk[0])
        print(f'K: {k}')


main()