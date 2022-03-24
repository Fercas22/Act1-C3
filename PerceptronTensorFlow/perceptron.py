import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
# fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)
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

def entrenamientoNeurona(epocas, tazaAprendizaje):
    X, Y = leerCSV()
    capa = tf.keras.layers.Dense(units = 1, input_shape = [1], activation='linear')
    modelo = tf.keras.Sequential([capa])
    # oculta1 = tf.keras.layers.Dense(units=3, input_shape=[1])
    # oculta2 = tf.keras.layers.Dense(units=3)
    # salida = tf.keras.layers.Dense(units=1)
    # modelo = tf.keras.Sequential([oculta1, oculta2, salida])

    modelo.compile(
        optimizer = tf.keras.optimizers.Adam(tazaAprendizaje),
        loss = 'mean_squared_error'
    )

    print('Comenzando entrenamiento...')
    historial = modelo.fit(X, Y, epochs=epocas, verbose=False)
    print('Modelo entrenado')

    plt.xlabel("Número de Epocas")
    plt.ylabel("Magnitud de perdida")
    plt.plot(historial.history["loss"])
    plt.show()

    print('Predicción!')
    resultado = modelo.predict([70])
    print(f'Resultado es --> {resultado}')

    # for i in range(len(X)):
    #     resultado = modelo.predict([X[i]])
    #     print(resultado)  



    # resultado = modelo.predict([89])
    # print("El resultado es " + str(resultado) + " fahrenheit!")
    # print(f'El resultado es --> {resultado}')

    print('Variables internas del modelo')
    print(capa.get_weights())

    # print(oculta1.get_weights())
    # print(oculta2.get_weights())
    # print(salida.get_weights())

# entrenamientoNeurona(2000, 0.1)