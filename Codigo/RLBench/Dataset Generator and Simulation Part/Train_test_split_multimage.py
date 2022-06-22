import os
from typing import Type
import pickle
import numpy as np
import csv
import re
import sys
import argparse
from random import sample
import shutil
import pandas as pd 
import glob
import sys

# Inicio
os.chdir("rlbench_data/reach_target")
path, dirs, files = next(os.walk(os.getcwd()))
print("VARIATIONS: " + str(len(dirs)))
VARIATIONS = len(dirs)
os.chdir("../../")

PATH_1 = "rlbench_data/reach_target/variation"
PATH_2 = "/episodes/"
TOTAL_DIRS = 0

for var in range(int(VARIATIONS)):
	PATH = PATH_1 + str(var) + PATH_2
	path, dirs, files = next(os.walk(PATH))
	TOTAL_DIRS = TOTAL_DIRS + len(dirs)
TRAIN_SIZE = round(len(dirs) * 0.7) if len(sys.argv) == 1 else round(len(dirs) * float(sys.argv[1])) # Por defecto el TRAIN_SIZE es un 70% del total
file_counter = 0
print(TRAIN_SIZE)
# Comprobacion de errores
if int(TRAIN_SIZE) >= len(dirs) or int(TRAIN_SIZE)  <= 0 : 
	print("Porcentaje de Train (" + str(sys.argv[1]) + ") erroneo.\nIntroduzca uno comprendido entre (0-1) sin incluir ni 0 ni 1.\nFinalizando ejecucion.")
	sys.exit()

# Generamos la particion aleatoria
datasets = list(range(len(dirs)))
train_datasets = sorted(sample([x for x in range(0,len(dirs))], int(TRAIN_SIZE)))
test_datasets = sorted(list(set(datasets) - set(train_datasets)))
print("Las " + str(TRAIN_SIZE * VARIATIONS) + " pruebas para Train son:\n" + str(train_datasets))
print("\nLas " + str(len(dirs)* VARIATIONS - int(TRAIN_SIZE * VARIATIONS)) + " pruebas para Test son:\n" + str(test_datasets))
print("Test dataset ordenado?: " + str(list(set(datasets) - set(train_datasets))))

# Preparo carpeta de Train
if not os.path.exists("Train_dataset"):
	os.makedirs("Train_dataset")
	os.makedirs("Train_dataset/Front_rgb")
	os.makedirs("Train_dataset/Left_rgb")
	os.makedirs("Train_dataset/Right_rgb")
	os.makedirs("Train_dataset/Overhead_rgb")
	os.makedirs("Train_dataset/Wrist_rgb")
else: 
	shutil.rmtree("Train_dataset") 
	os.makedirs("Train_dataset")
	os.makedirs("Train_dataset/Front_rgb")
	os.makedirs("Train_dataset/Left_rgb")
	os.makedirs("Train_dataset/Right_rgb")
	os.makedirs("Train_dataset/Overhead_rgb")
	os.makedirs("Train_dataset/Wrist_rgb")

# Preparo carpeta de Test
if not os.path.exists("Test_dataset"): 
	os.makedirs("Test_dataset") 
	os.makedirs("Test_dataset/Front_rgb")
	os.makedirs("Test_dataset/Left_rgb")
	os.makedirs("Test_dataset/Right_rgb")
	os.makedirs("Test_dataset/Overhead_rgb")
	os.makedirs("Test_dataset/Wrist_rgb")
else: 
	shutil.rmtree("Test_dataset") 
	os.makedirs("Test_dataset")
	os.makedirs("Test_dataset/Front_rgb")
	os.makedirs("Test_dataset/Left_rgb")
	os.makedirs("Test_dataset/Right_rgb")
	os.makedirs("Test_dataset/Overhead_rgb")
	os.makedirs("Test_dataset/Wrist_rgb")	

# Imagenes y Archivo de Acciones para Train
for var in range(int(VARIATIONS)):
	PATH = PATH_1 + str(var) + PATH_2
	for i in train_datasets:
		# Entramos en el directorio /episodes/episodeX
		path2 = PATH + "episode" + str(i) + "/front_rgb"
		path, dirs, files = next(os.walk(path2))
		#print("La carpeta episode" + str(i) + " tiene un total de " + str(len(files)) + " acciones")
		aux = file_counter
		file_counter = aux + int(len(files)) - 1
		aux_2 = 0
		#print("Imagenes [" + str(aux) + "-" + str(file_counter-1) + "]")
		
		# Imagenes
		for j in range(1,len(files)):
			# Front
			old_image = PATH + "episode" + str(i) + "/front_rgb/" + "" + str(j) + ".png"
			new_image = "Train_dataset/Front_rgb/" + str(aux + j - 1) + ".png"
			#shutil.copyfile(old_image, new_image)
			shutil.move(old_image, new_image) # Para mover en lugar de copiar
			
			# Left
			old_image = PATH + "episode" + str(i) + "/left_shoulder_rgb/" + "" + str(j) + ".png"
			new_image = "Train_dataset/Left_rgb/" + str(aux + j - 1) + ".png"
			#shutil.copyfile(old_image, new_image)
			shutil.move(old_image, new_image) # Para mover en lugar de copiar
			
			# Right
			old_image = PATH + "episode" + str(i) + "/right_shoulder_rgb/" + "" + str(j) + ".png"
			new_image = "Train_dataset/Right_rgb/" + str(aux + j - 1) + ".png"
			#shutil.copyfile(old_image, new_image)
			shutil.move(old_image, new_image) # Para mover en lugar de copiar
			
			# Overhead
			old_image = PATH + "episode" + str(i) + "/overhead_rgb/" + "" + str(j) + ".png"
			new_image = "Train_dataset/Overhead_rgb/" + str(aux + j - 1) + ".png"
			#shutil.copyfile(old_image, new_image)
			shutil.move(old_image, new_image) # Para mover en lugar de copiar
			
			# Depth
			old_image = PATH + "episode" + str(i) + "/wrist_rgb/" + "" + str(j) + ".png"
			new_image = "Train_dataset/Wrist_rgb/" + str(aux + j - 1) + ".png"
			#shutil.copyfile(old_image, new_image)
			shutil.move(old_image, new_image) # Para mover en lugar de copiar
			
			aux_2 = aux_2 + 1
		# Acciones
		# Velocidades
		old_action = PATH + "episode" + str(i) + "/Velocities.csv"
		new_action = "Train_dataset/" + str(var) + "_V_" + str(i) + ".csv"
		shutil.copyfile(old_action, new_action)
		# Posiciones
		old_action = PATH + "episode" + str(i) + "/Positions.csv"
		new_action = "Train_dataset/" + str(var) + "_P_" + str(i) + ".csv"
		shutil.copyfile(old_action, new_action)
# CSV para Train
filenames = []
for var in range(int(VARIATIONS)):
	for i in train_datasets:
		filenames.append("Train_dataset/" + str(var) + "_V_" + str(i) + ".csv")
print("Train csvs: "+str(filenames))
combined_csv = pd.concat(
   map(pd.read_csv, filenames), ignore_index=True)
#combined_csv = pd.concat([pd.read_csv(f) for f in filenames ], axis=0)
combined_csv.to_csv('Train_dataset/Train_Actions_V.csv', index=False)

filenames = []
for var in range(int(VARIATIONS)):
	for i in train_datasets:
		filenames.append("Train_dataset/" + str(var) + "_P_" + str(i) + ".csv") 
combined_csv = pd.concat(
   map(pd.read_csv, filenames), ignore_index=True)
#combined_csv = pd.concat([pd.read_csv(f) for f in filenames ], axis=0)
combined_csv.to_csv('Train_dataset/Train_Actions_P.csv', index=False)

print("\nLa longitud del fichero de Train es de:\n" + str(combined_csv.shape))


file_counter = 0

# Imagenes y Archivo de Acciones para Test
for var in range(int(VARIATIONS)):
	PATH = PATH_1 + str(var) + PATH_2
	for i in test_datasets:
		# Entramos en el directorio /episodes/episodeX
		path2 = PATH + "episode" + str(i) + "/front_rgb"
		path, dirs, files = next(os.walk(path2))
		#print("La carpeta episode" + str(i) + " tiene un total de " + str(len(files)) + " acciones")
		aux = file_counter
		file_counter = aux + int(len(files)) - 1
		aux_2 = 0
		#print("Imagenes [" + str(aux) + "-" + str(file_counter-1) + "]")
		
		# Imagenes
		for j in range(1,len(files)):
			# Front
			old_image = PATH + "episode" + str(i) + "/front_rgb/" + "" + str(j) + ".png"
			new_image = "Test_dataset/Front_rgb/" + str(aux + j - 1) + ".png"
			#shutil.copyfile(old_image, new_image)
			shutil.move(old_image, new_image) # Para mover en lugar de copiar
			
			# Left
			old_image = PATH + "episode" + str(i) + "/left_shoulder_rgb/" + "" + str(j) + ".png"
			new_image = "Test_dataset/Left_rgb/" + str(aux + j - 1) + ".png"
			#shutil.copyfile(old_image, new_image)
			shutil.move(old_image, new_image) # Para mover en lugar de copiar
			
			# Right
			old_image = PATH + "episode" + str(i) + "/right_shoulder_rgb/" + "" + str(j) + ".png"
			new_image = "Test_dataset/Right_rgb/" + str(aux + j - 1) + ".png"
			#shutil.copyfile(old_image, new_image)
			shutil.move(old_image, new_image) # Para mover en lugar de copiar
			
			# Overhead
			old_image = PATH + "episode" + str(i) + "/overhead_rgb/" + "" + str(j) + ".png"
			new_image = "Test_dataset/Overhead_rgb/" + str(aux + j - 1) + ".png"
			#shutil.copyfile(old_image, new_image)
			shutil.move(old_image, new_image) # Para mover en lugar de copiar
			
			# Depth
			old_image = PATH + "episode" + str(i) + "/wrist_rgb/" + "" + str(j) + ".png"
			new_image = "Test_dataset/Wrist_rgb/" + str(aux + j - 1) + ".png"
			#shutil.copyfile(old_image, new_image)
			shutil.move(old_image, new_image) # Para mover en lugar de copiar
			
			aux_2 = aux_2 + 1
			
		# Acciones
		# Velocidades
		old_action = PATH + "episode" + str(i) + "/Velocities.csv"
		new_action = "Test_dataset/" + str(var) + "_V_" + str(i) + ".csv"
		shutil.copyfile(old_action, new_action)
		# Posiciones
		old_action = PATH + "episode" + str(i) + "/Positions.csv"
		new_action = "Test_dataset/" + str(var) + "_P_" + str(i) + ".csv"
		shutil.copyfile(old_action, new_action)
# CSV para Test
filenames = []
for var in range(int(VARIATIONS)):
	for i in test_datasets:
		filenames.append("Test_dataset/" + str(var) + "_V_" + str(i) + ".csv") 
print("Test csvs: "+str(filenames))
combined_csv = pd.concat(
   map(pd.read_csv, filenames), ignore_index=True)
#combined_csv = pd.concat([pd.read_csv(f) for f in filenames ], axis=0)
combined_csv.to_csv('Test_dataset/Test_Actions_V.csv', index=False)

filenames = []
for var in range(int(VARIATIONS)):
	for i in test_datasets:
		filenames.append("Test_dataset/" + str(var) + "_P_" + str(i) + ".csv") 
combined_csv = pd.concat(
   map(pd.read_csv, filenames), ignore_index=True)
#combined_csv = pd.concat([pd.read_csv(f) for f in filenames ], axis=0)
combined_csv.to_csv('Test_dataset/Test_Actions_P.csv', index=False)

print("\nLa longitud del fichero de Test es de:\n" + str(combined_csv.shape))

file_counter = 0

# Borramos los datos innecesarios
print(os.getcwd())
shutil.rmtree("rlbench_data") 
