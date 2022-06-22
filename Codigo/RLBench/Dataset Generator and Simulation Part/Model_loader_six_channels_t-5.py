import os
import sys
import csv
from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget
import numpy as np
import pandas as pd
import pickle
import shutil
import torch
from PIL import Image
from torchvision import models
from torchvision import transforms
from torch import nn
from absl import app
from absl import flags
from pyrep.const import RenderMode
from rlbench.backend.utils import task_file_to_task_class
import rlbench.backend.task as task
from rlbench.backend.const import *
from rlbench.backend import utils
import joblib
from sklearn.preprocessing import MinMaxScaler

img_size = [128, 128]
NORM = sys.argv[1] if len(sys.argv) > 1 else 0
NUM_PRUEBAS = sys.argv[2] if len(sys.argv) > 2 else 10
NUM_ITERS = sys.argv[3] if len(sys.argv) > 3 else 100
SAVE = sys.argv[4] if len(sys.argv) > 4 else 1
OPTION = sys.argv[5] if len(sys.argv) > 5 else False
MODEL_NAME = sys.argv[6] if len(sys.argv) > 6 else 'model_pytorch_0014' 

if OPTION != True and OPTION != False:
	OPTION = True

print('Parametros introducidos: \n -NORM: ', str(NORM), '\n -NUM_PRUEBAS: ', str(NUM_PRUEBAS), '\n -NUM_ITERS: ', str(NUM_ITERS), '\n -OPTION: ', str(OPTION), '\n -SAVE: ', str(SAVE), '\n -MODEL: ', str(MODEL_NAME), '\n')

# Preparo carpeta de Train
if not os.path.exists("Imagenes GIF"):
	os.makedirs("Imagenes GIF")
	os.makedirs("Imagenes GIF/Front")
	os.makedirs("Imagenes GIF/Left")
	os.makedirs("Imagenes GIF/Right")
	os.makedirs("Imagenes GIF/Overhead")
	os.makedirs("Imagenes GIF/Wrist")
else: 
	shutil.rmtree("Imagenes GIF")
	os.makedirs("Imagenes GIF")
	os.makedirs("Imagenes GIF/Front")
	os.makedirs("Imagenes GIF/Left")
	os.makedirs("Imagenes GIF/Right")
	os.makedirs("Imagenes GIF/Overhead")
	os.makedirs("Imagenes GIF/Wrist")
	
# Preparo la carpeta de Resultados
if not os.path.exists("Resultados"):
	os.makedirs("Resultados")

# PREDICT
class MultiImage(nn.Module):
    def __init__(self, fe, clf):
        super(MultiImage, self).__init__()
        self.fe = fe
        self.clf = clf
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
    def forward(self, x):

    	x1, x2, x3, x4, x5, x6 = x

    	x2 = torch.from_numpy(x2)
    	x2 = x2.unsqueeze(0)
    	x2 = x2.to(torch.float32)
    	
    	
    	x3 = torch.from_numpy(x3)
    	x3 = x3.unsqueeze(0)
    	x3 = x3.to(torch.float32)
    	
    	x4 = torch.from_numpy(x4)
    	x4 = x4.unsqueeze(0)
    	x4 = x4.to(torch.float32)
    	
    	x5 = torch.from_numpy(x5)
    	x5 = x5.unsqueeze(0)
    	x5 = x5.to(torch.float32)
    	
    	x6 = torch.from_numpy(x6)
    	x6 = x6.unsqueeze(0)
    	x6 = x6.to(torch.float32)

    	
    	f1 = self.fe(x1)
    	f = self.flatten(self.avg_pool(f1))
    	
    	f = torch.cat((f, x2, x3, x4, x5, x6), dim=1)
    	return self.clf(f)

def predict(model, obs, obs_prev, action_prev, action_prev2, action_prev3, action_prev4, action_prev5, i, SAVE):
    model = model.eval()
    # Get image from observation
    image, front, overhead, left, right, wrist = get_image_from_observation(obs, obs_prev, i)
    
    if int(SAVE) == 1:
	    # Save three first Images
	    name_1 = "Imagenes GIF/Front/" + str(i) + ".png"
	    name_2 = "Imagenes GIF/Right/" + str(i) + ".png"
	    name_3 = "Imagenes GIF/Left/" + str(i) + ".png"
	    name_4 = "Imagenes GIF/Overhead/" + str(i) + ".png"
	    name_5 = "Imagenes GIF/Wrist/" + str(i) + ".png"

	    front.save(name_1, "PNG")
	    right.save(name_2, "PNG")
	    left.save(name_3, "PNG")
	    overhead.save(name_4, "PNG")
	    wrist.save(name_5, "PNG")
    
    transformations = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406, 0.485, 0.456, 0.406, 0.485, 0.456, 0.406, 0.485, 0.456, 0.406, 0.485, 0.456, 0.406), (0.229, 0.224, 0.225, 0.229, 0.224, 0.225, 0.229, 0.224, 0.225, 0.229, 0.224, 0.225, 0.229, 0.224, 0.225))])
    
    image = transformations(image).unsqueeze(dim=0)

    with torch.no_grad():
        pred = model((image, action_prev, action_prev2, action_prev3, action_prev4, action_prev5))
    pred = pred.detach().cpu().numpy()
    prediction = pred[0, :].tolist()
    prediction.append(1)
    #print("PRED", prediction, " FIN")
    return prediction
    
def get_image_from_observation(obs, obs_prev, i):
	im1 = Image.fromarray(obs.front_rgb)
	im2 = Image.fromarray(obs.overhead_rgb)
	im3 = Image.fromarray(obs.left_shoulder_rgb)
	im4 = Image.fromarray(obs.right_shoulder_rgb)
	im5 = Image.fromarray(obs.wrist_rgb)

	if ((i % int(NUM_ITERS) == 0) or (i == 0)):
		im6 = im1
		im7 = im2
		im8 = im3
		im9 = im4
		im10 = im5
		
	else :
		im6 = Image.fromarray(obs_prev.front_rgb)
		im7 = Image.fromarray(obs_prev.overhead_rgb)
		im8 = Image.fromarray(obs_prev.left_shoulder_rgb)
		im9 = Image.fromarray(obs_prev.right_shoulder_rgb)
		im10 = Image.fromarray(obs_prev.wrist_rgb)
		
		
	im_actual = np.concatenate((im1, im2, im3, im4, im5), axis=2)
	im_previa_1 = np.concatenate((im6, im7, im8, im9, im10), axis=2)
	image = np.concatenate((im_actual, im_previa_1), axis=1)	

	return image, im1, im2, im3, im4, im5 # Le pasamos la im1 para que la guarde y haga GIFs en predict()

obs_config = ObservationConfig()
obs_config.set_all(True)
obs_config.right_shoulder_camera.image_size = img_size
obs_config.left_shoulder_camera.image_size = img_size
obs_config.overhead_camera.image_size = img_size
obs_config.wrist_camera.image_size = img_size
obs_config.front_camera.image_size = img_size

action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)

env = Environment(
    action_mode = action_mode,
    obs_config=obs_config,
headless=OPTION)
env.launch()

task = env.get_task(ReachTarget)
print("TASK ", task)
print("TARGET ", ReachTarget)

# Load model
model = torch.load(MODEL_NAME, map_location ='cpu')

#obs = env.reset()
obs = None
obs_prev = None
terminate = False
Done = False
i = 0
its = 0
finalize = 0
vector_init = [] 
vector_finalize = [] 
vector_finalize_bool = []

scaler = joblib.load("MinMax_scaler.save") 
num_iters_seguidas = 0

# Condicion de inicio
print('Reset Episode')	
print('Acciones desde t hasta t-5:')
action_prev = np.array([0,0,0,0,0,0,0], dtype="float32")
action_prev2 = np.array([0,0,0,0,0,0,0], dtype="float32")
action_prev3 = np.array([0,0,0,0,0,0,0], dtype="float32")
action_prev4 = np.array([0,0,0,0,0,0,0], dtype="float32")
action_prev5 = np.array([0,0,0,0,0,0,0], dtype="float32")

vector_init.append(i)

if (terminate == False):
	vector_finalize_bool.append(0)
else:
	vector_finalize_bool.append(1)

terminate = False
descriptions, obs = task.reset()
its = its + 1
obs_prev = obs

while (not terminate) or (int(NUM_PRUEBAS) > int(its)) :
	print("Reinicio cuando = 0 -> ", str(int(NUM_ITERS) - int(num_iters_seguidas)))
	print("Pruebas realizadas correctamente: " , int(finalize) , "/" , int(NUM_PRUEBAS))
	if ((int(NUM_ITERS) - int(num_iters_seguidas) == 0) or (terminate == True)):
	
		print('Reset Episode')
		print('Acciones desde t hasta t-5:')
		action_prev = np.array([0,0,0,0,0,0,0], dtype="float32")
		action_prev2 = np.array([0,0,0,0,0,0,0], dtype="float32")
		action_prev3 = np.array([0,0,0,0,0,0,0], dtype="float32")
		action_prev4 = np.array([0,0,0,0,0,0,0], dtype="float32")
		action_prev5 = np.array([0,0,0,0,0,0,0], dtype="float32")
		print(action_prev)
		print(action_prev2)
		print(action_prev3)
		print(action_prev4)
		print(action_prev5)
		
		num_iters_seguidas = 0
		
		vector_init.append(i)
		
		if (terminate == False):
			vector_finalize_bool.append(0)
		else:
			vector_finalize_bool[-1] = 1
		
		terminate = False
		descriptions, obs = task.reset()
		its = its + 1
		obs_prev = obs
	action = predict(model, obs, obs_prev, action_prev, action_prev2, action_prev3, action_prev4, action_prev5, i, SAVE)
	action_no_norm = np.array(action, dtype="float32")
	
	if NORM == str(1): # En caso de tener que desnormalizar las acciones 
		#print(action)
		action = np.array(action[:-1])
		action = action[np.newaxis, :]
		action = scaler.inverse_transform(action)
		#print(action)
		action = np.append(action, 1)

	obs_prev = obs
	obs, reward, terminate = task.step(action)
	action_prev5 = action_prev4
	action_prev4 = action_prev3
	action_prev3 = action_prev2
	action_prev2 = action_prev
	action_prev = action_no_norm[:-1]
	
	print('Acciones desde t hasta t-5:')
	print(action_prev)
	print(action_prev2)
	print(action_prev3)
	print(action_prev4)
	print(action_prev5)
	
	i = i + 1
	num_iters_seguidas = num_iters_seguidas + 1
	
	if terminate == True:
		finalize = finalize + 1
		num_iters_seguidas = 0
	
	print('\nIteracion: ' , i)
	print('Iteraciones seguidas: ' , num_iters_seguidas)
	print("ITS: ", int(its))
	print("Num Pruebas: ", int(NUM_PRUEBAS))
	
	if its > int(NUM_PRUEBAS):
		terminate = True
		
	print('Terminate :', terminate)

vector_init.append(99)
vector_pruebas = []
header = ['Inicio','Final','Intentos','Finaliza']

for i in range(int(NUM_PRUEBAS)+1):
	vector_pruebas.append(vector_init[i+1] - vector_init[i])		

vector_pruebas[-1] = 999
vector_init[-1] = 888
vector_finalize_bool[-1] = 777
CSV_NAME = str(MODEL_NAME) + ".csv"
PATH = 'Resultados/' + str(CSV_NAME)


for i in range(len(vector_init)-1):
	vector_finalize.append(int(vector_init[i+1])-1)
	vector_pruebas.append(int(vector_finalize[i]) - int(vector_init[i]))


with open(CSV_NAME, 'w') as f:
     write = csv.writer(f)
     write.writerow(header)
     for i in range(len(vector_init)-2):
             row = []
             row.append(vector_init[i])
             row.append(vector_finalize[i])
             row.append(vector_pruebas[i])
             row.append(vector_finalize_bool[i])
             write.writerow(row)

'''


with open(CSV_NAME, 'w') as f:
      
    # using csv.writer method from CSV package
    write = csv.writer(f)
      
    write.writerow(vector_finalize[:-1])
    write.writerow(vector_pruebas)
    write.writerow(vector_finalize_bool[:-1])
 '''
shutil.move(CSV_NAME, PATH) 
		
print("Done!")
env.shutdown()
