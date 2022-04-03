import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as date

predicciones = []

def leerCSV():
    X = []
    Y = []

    dataset = pd.read_csv('./datos.csv')
    for i in range(len(dataset)):
        X.append(dataset['X1'][i])
        Y.append(dataset['Y'][i])

    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)

    return X, Y

X, Y = leerCSV()

def entrenamientoNeurona(epocas, tasaAprendizaje):
   
    capa = tf.keras.layers.Dense(units = 1, input_shape = [1], activation='linear')
    modelo = tf.keras.Sequential([capa])
    # oculta1 = tf.keras.layers.Dense(units=3, input_shape=[1])
    # oculta2 = tf.keras.layers.Dense(units=3)
    # salida = tf.keras.layers.Dense(units=1)
    # modelo = tf.keras.Sequential([oculta1, oculta2, salida])

    modelo.compile(
        optimizer = tf.keras.optimizers.Adam(tasaAprendizaje),
        loss = 'mean_squared_error'
    )

    print('Comenzando entrenamiento...')
    historial = modelo.fit(X, Y, epochs=epocas, verbose=True)
    print('Modelo entrenado')
    # return capa, modelo, historial
    prediccion(modelo, capa)
    grafica(historial)
    


def grafica(historial):
    plt.xlabel("Número de Epocas")
    plt.ylabel("Magnitud de perdida")
    plt.plot(historial.history["loss"])


    tiempo = str(date.datetime.now()).replace(':','_')
    tiempo = tiempo.replace('-','_')
    tiempo = tiempo.replace(' ','__')
    print(tiempo[:19])
    plt.savefig('./grafica/grafica'+'_'+tiempo[:19])
    plt.show()


def prediccion(modelo, capa):
    for i in range(len(X)):
        resultado = modelo.predict([X[i]])
        print(resultado)
        print('Variables internas del modelo')

    # resultado = modelo.predict([89])
    # print("El resultado es " + str(resultado) + " fahrenheit!")
    # print(f'El resultado es --> {resultado}')

    

    # print(oculta1.get_weights())
    # print(oculta2.get_weights())
    # print(salida.get_weights())
    print(capa.get_weights())
    