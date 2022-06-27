import os
import sys
import csv
import numpy as np
import pandas as pd
import pickle
import shutil
import matplotlib.pyplot as plt
from pathlib import Path
import imageio

MODEL = sys.argv[1] if len(sys.argv) > 1 else 0

if MODEL == 0:
	print('Error en el modelo. No existente.')

path, dirs, files = next(os.walk(MODEL))
print(dirs)
for dir in dirs:
    print("\n", os.getcwd())
    os.chdir(MODEL + '/' + dir)
    path, dirs, files = next(os.walk('./'))
    print(path)
    # Comenzamos limpiando las imagenes y gifs anteriores dejando solo los CSV
    for filename in Path(path).glob("*.gif"):
        filename.unlink()
    for filename in Path(path).glob("*.png"):
        filename.unlink()

    path, dirs, files = next(os.walk('./')) # 1: CSV resultados, 2: CSV predicciones

    # Primero voy a guardarme la columna de intentos del CSV de resultados
    intentos = []
    with open(files[0]) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count != 0: # Nos saltamos el Header
                intentos.append(int(row[2]))
            line_count += 1
        print(intentos)

    # Tras obtener los intentos, empezamos a plotear por ellos leyendo el CSV de predicciones
    inicios = [0]
    y1 = []
    y2 = []
    y3 = []
    y4 = []
    y5 = []
    y6 = []
    y7 = []
    with open(files[1]) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count != 0: # Nos saltamos el Header
                if row[0] == '-':
                    inicios.append(line_count)
                    row[0] = 99
                    row[1] = 99
                    row[2] = 99
                    row[3] = 99
                    row[4] = 99
                    row[5] = 99
                    row[6] = 99
                y1.append(float(row[0]))
                y2.append(float(row[1]))
                y3.append(float(row[2]))
                y4.append(float(row[3]))
                y5.append(float(row[4]))
                y6.append(float(row[5]))
                y7.append(float(row[6]))
            line_count += 1

    print(inicios)

    # Preparamos los datos para plotearlos
    for i in range(len(intentos)):
        # Preparamos el Eje X
        X = np.arange(1,intentos[i]+1)

        # Preparamos los Eje Y
        Y1 = y1[inicios[i]:inicios[i+1]-1]
        Y2 = y2[inicios[i]:inicios[i+1]-1]
        Y3 = y3[inicios[i]:inicios[i+1]-1]
        Y4 = y4[inicios[i]:inicios[i+1]-1]
        Y5 = y5[inicios[i]:inicios[i+1]-1]
        Y6 = y6[inicios[i]:inicios[i+1]-1]
        Y7 = y7[inicios[i]:inicios[i+1]-1]

        plt.figure()
        plt.plot(X, Y1, label="Art1")
        plt.plot(X, Y2, label="Art2")
        plt.plot(X, Y3, label="Art3")
        plt.plot(X, Y4, label="Art4")
        plt.plot(X, Y5, label="Art5")
        plt.plot(X, Y6, label="Art6")
        plt.plot(X, Y7, label="Art7")
        plt.legend()
        plt.savefig('Prediction_' + str(i) + '.png')

    # Hacemos un GIF para mostrar las diferencias entre las predicciones
    images = []
    path, dirs, files = next(os.walk('./'))
    print(files)
    filenames = []
    for f in files:
        if '.png' in f:
            filenames.append(f)
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave('Prediction.gif', images, duration=1)

    os.chdir('../../')


