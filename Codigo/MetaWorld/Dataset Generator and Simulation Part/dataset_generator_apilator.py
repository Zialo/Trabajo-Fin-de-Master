import metaworld
import random
import matplotlib.pyplot as plt
import numpy as np
import csv
from metaworld.policies.sawyer_pick_out_of_hole_v2_policy import SawyerPickOutOfHoleV2Policy
import tqdm
import os
import random
import shutil
import sys
from PIL import Image
import pandas as pd
import imageio as iio
from pathlib import Path
from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
                            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)

NAME = 'Dataset_Mujoco'
carpetas = ['MuJoCo_100_1',
            'MuJoCo_100_2',
            'MuJoCo_100_3',
            'MuJoCo_100_4',
            'MuJoCo_100_5',
            'MuJoCo_500',
            'MuJoCo_500_2',
            'MuJoCo_600']

csv_train = ['MuJoCo_100_1/Train_dataset/Train_Actions.csv',
             'MuJoCo_100_2/Train_dataset/Train_Actions.csv',
             'MuJoCo_100_3/Train_dataset/Train_Actions.csv',
             'MuJoCo_100_4/Train_dataset/Train_Actions.csv',
             'MuJoCo_100_5/Train_dataset/Train_Actions.csv',
             'MuJoCo_500/Train_dataset/Train_Actions.csv',
             'MuJoCo_500_2/Train_dataset/Train_Actions.csv',
             'MuJoCo_600/Train_dataset/Train_Actions.csv']

csv_test = ['MuJoCo_100_1/Test_dataset/Test_Actions.csv',
            'MuJoCo_100_2/Test_dataset/Test_Actions.csv',
            'MuJoCo_100_3/Test_dataset/Test_Actions.csv',
            'MuJoCo_100_4/Test_dataset/Test_Actions.csv',
            'MuJoCo_100_5/Test_dataset/Test_Actions.csv',
            'MuJoCo_500/Test_dataset/Test_Actions.csv',
            'MuJoCo_500_2/Test_dataset/Test_Actions.csv',
            'MuJoCo_600/Test_dataset/Test_Actions.csv']

# Genero las carpetas
if not os.path.exists(NAME):
    os.makedirs(NAME + "/")
    os.makedirs(NAME + "/Test_dataset")
    os.makedirs(NAME + "/Test_dataset/Gripper")
    os.makedirs(NAME + "/Test_dataset/Corner")
    os.makedirs(NAME + "/Test_dataset/Corner2")
    os.makedirs(NAME + "/Test_dataset/Corner3")
    os.makedirs(NAME + "/Test_dataset/Top")
    os.makedirs(NAME + "/Test_dataset/BehindGripper")
    os.makedirs(NAME + "/Train_dataset")
    os.makedirs(NAME + "/Train_dataset/Gripper")
    os.makedirs(NAME + "/Train_dataset/Corner")
    os.makedirs(NAME + "/Train_dataset/Corner2")
    os.makedirs(NAME + "/Train_dataset/Corner3")
    os.makedirs(NAME + "/Train_dataset/Top")
    os.makedirs(NAME + "/Train_dataset/BehindGripper")
else:
    shutil.rmtree(NAME)
    os.makedirs(NAME + "/")
    os.makedirs(NAME + "/Test_dataset")
    os.makedirs(NAME + "/Test_dataset/Gripper")
    os.makedirs(NAME + "/Test_dataset/Corner")
    os.makedirs(NAME + "/Test_dataset/Corner2")
    os.makedirs(NAME + "/Test_dataset/Corner3")
    os.makedirs(NAME + "/Test_dataset/Top")
    os.makedirs(NAME + "/Test_dataset/BehindGripper")
    os.makedirs(NAME + "/Train_dataset")
    os.makedirs(NAME + "/Train_dataset/Gripper")
    os.makedirs(NAME + "/Train_dataset/Corner")
    os.makedirs(NAME + "/Train_dataset/Corner2")
    os.makedirs(NAME + "/Train_dataset/Corner3")
    os.makedirs(NAME + "/Train_dataset/Top")
    os.makedirs(NAME + "/Train_dataset/BehindGripper")

file_counter = 0
file_counter_test = 0

for carpeta in carpetas:

    # Imagenes de Train
    path_train = carpeta + '/Train_dataset/'
    path_aux = path_train + '/Gripper'
    path, dirs, files = next(os.walk(path_aux))
    print(path)
    print(len(files))

    for j in range(len(files)):
        # Gripper
        old_image = path_train + "/Gripper/" + str(j) + ".png"
        new_image = NAME + "/Train_dataset/Gripper/" + str(file_counter) + ".png"
        shutil.copyfile(old_image, new_image)

        # Corner
        old_image = path_train + "/Corner/" + str(j) + ".png"
        new_image = NAME + "/Train_dataset/Corner/" + str(file_counter) + ".png"
        shutil.copyfile(old_image, new_image)

        # Corner2
        old_image = path_train + "/Corner2/" + str(j) + ".png"
        new_image = NAME + "/Train_dataset/Corner2/" + str(file_counter) + ".png"
        shutil.copyfile(old_image, new_image)

        # Corner3
        old_image = path_train + "/Corner3/" + str(j) + ".png"
        new_image = NAME + "/Train_dataset/Corner3/" + str(file_counter) + ".png"
        shutil.copyfile(old_image, new_image)

        # Top
        old_image = path_train + "/Top/" + str(j) + ".png"
        new_image = NAME + "/Train_dataset/Top/" + str(file_counter) + ".png"
        shutil.copyfile(old_image, new_image)

        # BehindGripper
        old_image = path_train + "/BehindGripper/" + str(j) + ".png"
        new_image = NAME + "/Train_dataset/BehindGripper/" + str(file_counter) + ".png"
        shutil.copyfile(old_image, new_image)

        file_counter += 1
        if file_counter % 100 == 0:
            print(file_counter)
            print(old_image, " > ", new_image)

    # CSV para Train
    print('Empieza CSV Train')
    action_train = []
    for i in csv_train:
        with open(i) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count in range(len(files)):
                    action_train.append(row)

    # Genero un nuevo CSV de Train
    with open(NAME + "/Train_dataset/Train_Actions.csv", 'w') as f:
        write = csv.writer(f)
        for i_a in range(len(action_train)):
            row = []
            for a in range(4):
                row.append(action_train[i_a][a])
            write.writerow(row)

    # Imagenes de Test
    path_test = carpeta + '/Test_dataset/'
    path_aux = path_test + '/Gripper'
    path, dirs, files = next(os.walk(path_aux))
    print(path)
    print(len(files))

    for j in range(len(files)):
        # Gripper
        old_image = path_test + "/Gripper/" + str(j) + ".png"
        new_image = NAME + "/Test_dataset/Gripper/" + str(file_counter_test) + ".png"
        shutil.copyfile(old_image, new_image)

        # Corner
        old_image = path_test + "/Corner/" + str(j) + ".png"
        new_image = NAME + "/Test_dataset/Corner/" + str(file_counter_test) + ".png"
        shutil.copyfile(old_image, new_image)

        # Corner2
        old_image = path_test + "/Corner2/" + str(j) + ".png"
        new_image = NAME + "/Test_dataset/Corner2/" + str(file_counter_test) + ".png"
        shutil.copyfile(old_image, new_image)

        # Corner3
        old_image = path_test + "/Corner3/" + str(j) + ".png"
        new_image = NAME + "/Test_dataset/Corner3/" + str(file_counter_test) + ".png"
        shutil.copyfile(old_image, new_image)

        # Top
        old_image = path_test + "/Top/" + str(j) + ".png"
        new_image = NAME + "/Test_dataset/Top/" + str(file_counter_test) + ".png"
        shutil.copyfile(old_image, new_image)

        # BehindGripper
        old_image = path_test + "/BehindGripper/" + str(j) + ".png"
        new_image = NAME + "/Test_dataset/BehindGripper/" + str(file_counter_test) + ".png"
        shutil.copyfile(old_image, new_image)

        file_counter_test += 1
        if file_counter_test % 100 == 0:
            print(file_counter_test)
            print(old_image, " > ", new_image)

    # CSV para Test
    print('Empieza CSV Test')
    action_test = []
    for i in csv_test:
        with open(i) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count in range(len(files)):
                    action_test.append(row)

    # Genero un nuevo CSV de Test
    with open(NAME + "/Test_dataset/Test_Actions.csv", 'w') as f:
        write = csv.writer(f)
        for i_a in range(len(action_test)):
            row = []
            for a in range(4):
                row.append(action_test[i_a][a])
            write.writerow(row)
