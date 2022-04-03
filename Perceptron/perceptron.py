import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import datetime as date

def dataset():

    dataset = pd.read_csv('./datos.csv')
    X = []
    Y = []

    for i in range(len(dataset)):
        X.append([1, dataset['X1'][i]])
        Y.append(dataset['Y'][i])
    
    return X, Y



def entranamiento_Neurona(n, wk):
    X, Y = dataset()
    X = np.array(X).transpose()
    Y = np.array(Y)
    
    k = 0

    errores = []
    generaciones = []
   
    while(True):
        
        k += 1
        uk = np.dot(wk, X)

        yck = uk
        ek = Y - yck
        #V
        # temp = np.dot(X, ek) * n

        errorM = max(ek)
        ek = ek.transpose()

        temp = np.dot(X, ek) * n

        wt = wk + temp

        cont = 0

        for i in range(len(ek)):
            cont += ek[i]**2

        wk = wt
        errores.append((math.sqrt(cont)))
        generaciones.append(k)
        print(f"Error Máximo {errorM}")

        if errorM < 0.1 or np.isnan(errorM) or np.isinf(errorM):
            print(errorM)
            # print(f'pesos: {wk}')
            print(f'yck {yck}')
            # for i in yck:
            print(wk)
            print('GENERACIONES')
            print(k)
            return errores,generaciones,wk



def entrada(wk, ns):
    curvas = []
    for i in range(len(ns)):
        curvas.append(entranamiento_Neurona(ns[i], wk))
    
    grafica(ns, curvas)

def grafica(ns, curvas):

    grafica = plt.subplot(1, 2, 1)
    
    grafica.set_title('Gráfica')
    
    for x in range(len(curvas)):
        marker = 'o'
        grafica.plot(curvas[x][1], curvas[x][0], marker=f'{marker}', label=f'η{x+1} = {ns[x]}')
    grafica.set_xlabel('Iteraciones')
    grafica.set_ylabel('Magnitud de Perdida')
    grafica.legend()
    
    
    tabla = plt.subplot(1, 2, 2)
    tabla.axis('tight')
    tabla.axis('off')
    tabla.set_title('Tabla')
    table = [['η (Tasa de aprendizaje)', 'Ultimos pesos de W']]
    
    for y in range(len(ns)):
       
        redo = []
        for f in range(len(curvas[y][2])):
            redo.append(round(curvas[y][2][f],3))

        table.append([ns[y], redo])

    table = tabla.table(cellText=table, loc='center', cellLoc= 'center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 3)

    plt.tight_layout()

    tiempo = str(date.datetime.now()).replace(':','_')
    tiempo = tiempo.replace('-','_')
    tiempo = tiempo.replace(' ','__')
    print(tiempo[:19])
    plt.savefig('./grafica/grafica'+tiempo[:19])
    
    plt.show()

# entrada([0.45, 0.10],[0.00001, 0.000001, 0.0000001, 0.00000001])
# entrada([0.20, 0.45], [0.0000001, 0.00000001, 0.0000000001])
# entrada([0.35, 0.40],[0.000001, 0.0000001])
# entrada([0.20,0.45], [0.000001])
# entrada([0.20,0.45], [0.0000001])
# entrada([0.20,0.45], [0.00000001])