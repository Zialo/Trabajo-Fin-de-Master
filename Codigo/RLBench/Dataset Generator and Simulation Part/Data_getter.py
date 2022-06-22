import os
from typing import Type
import pickle
import numpy as np
import csv
import re
import sys

TASK = sys.argv[2] if len(sys.argv) > 2 else 'reach_target'

os.chdir("rlbench_data/" + TASK) 
path, dirs, files = next(os.walk(os.getcwd()))
print("VARIATIONS: " + str(len(dirs)))
VARIATIONS = len(dirs)
os.chdir("../../")

PATH_1 = "rlbench_data/" + TASK + "/variation"
PATH_2 = "/episodes/"
for var in range(int(VARIATIONS)):
	PATH = PATH_1 + str(var) + PATH_2
	path, dirs, files = next(os.walk(PATH))
	os.chdir(PATH)

	for i in range(len(dirs)):
		path = "episode" + str(i)
		os.chdir(path)

		with open("low_dim_obs.pkl", "rb") as f :
			pkl = pickle.load(f)
			
		print("Dimension: ", len(pkl))
		
		# Velocidades
		f1 = open("Velocities.csv", "w+")
		writer = csv.writer(f1)
		writer.writerow(["X1", "X2", "X3", "X4", "X5", "X6", "X7"]) # Headers

		for i in range(1,len(pkl)): 
			obj = pkl[i]
			writer.writerow([str(obj.joint_velocities[0]), str(obj.joint_velocities[1]), str(obj.joint_velocities[2]), str(obj.joint_velocities[3]), str(obj.joint_velocities[4]), str(obj.joint_velocities[5]), str(obj.joint_velocities[6])])

	
		f1.close()	
		
		# Posiciones
		f2 = open("Positions.csv", "w+")
		writer = csv.writer(f2)
		writer.writerow(["X1", "X2", "X3", "X4", "X5", "X6", "X7"]) # Headers

		for i in range(1,len(pkl)): 
			obj = pkl[i]
			writer.writerow([str(obj.joint_positions[0]), str(obj.joint_positions[1]), str(obj.joint_positions[2]), str(obj.joint_positions[3]), str(obj.joint_positions[4]), str(obj.joint_positions[5]), str(obj.joint_positions[6])])

	
		f2.close()
		
		os.chdir("../")
	os.chdir("../../../../")
	print("Done")
