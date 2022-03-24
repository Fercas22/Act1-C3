from cmath import nan
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd

def dataset():

    dataset = pd.read_csv('./datos.csv')
    X = []
    Y = []

    for i in range(len(dataset)):
        X.append([1, dataset['X1'][i]])
        Y.append(dataset['Y'][i])
    
    return X, Y


def entranamiento_Neurona(n, wk, epocas):

    X, Y = dataset()

    k = 0

    errores = []
    generaciones = []
    # print(X)
    # print(Y)
    # print(n)
    # print(wk)
    # ek = np.array()
    while(True):
        
        k += 1
        print("Epoca: ", k)
        #V
        # uk = np.dot(wk, X)
        uk = np.dot(X, wk)

        yck = uk
        ek = Y - yck
        #V
        # temp = np.dot(X, ek) * n
        temp = np.dot(ek, X) * n

        wt = wk + temp

        cont = 0

        for i in range(len(ek)):
            cont += ek[i]**2

        wk = wt
        errores.append((math.sqrt(cont)))
        generaciones.append(k)
    

        if np.all(yck == Y):
            return errores, generaciones, wk

        if epocas == k:
            return errores, generaciones, wk



def entrada(wk, ns, epocas):
    curvas = []
    for i in range(len(ns)):
        curvas.append(entranamiento_Neurona(ns[i], wk, epocas))
    # figure2 = plt.figure(figsize=(15, 7))
    grafica(ns, curvas)

def grafica(ns, curvas):

    ax = plt.subplot(1, 2, 1)
    df = pd.DataFrame(curvas)
    ax.set_title('Gráfica')
    # curvas = np.fillna(0)
    
    for x in range(len(curvas)):
        marker = 'o'
        ax.plot(curvas[x][1], curvas[x][0],
                marker=f'{marker}', label=f'η{x+1} = {ns[x]}')

    ax.legend()
    
    
    ax2 = plt.subplot(1, 2, 2)
    ax2.axis('tight')
    ax2.axis('off')

    table = [['η (Taza de aprendizaje)', 'Ultimos pesos de W']]
    
    for y in range(len(ns)):
       
        redo = []
        for f in range(len(curvas[y][2])):
            
            if np.isnan(curvas[y][2][f]):
                print(curvas[y][2][f])
                print('Esto es el los NaN')
                df.fillna(0)
                redo.append(0)
            else:
                redo.append(round(curvas[y][2][f], 3))
            # redo.append(round(curvas[y][2][f], 3))

        table.append([ns[y], redo])

    table = ax2.table(cellText=table, loc='center', cellLoc= 'center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 3)

    plt.tight_layout()
    plt.show()