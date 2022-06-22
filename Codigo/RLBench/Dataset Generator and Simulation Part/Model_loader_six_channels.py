import os
import sys
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

# PREDICT
class MultiImage(nn.Module):
    def __init__(self, fe, clf):
        super(MultiImage, self).__init__()
        self.fe = fe
        self.clf = clf
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
    def forward(self, x):

    	x1, x2 = x

    	x2 = torch.from_numpy(x2)
    	x2 = x2.unsqueeze(0)
    	x2 = x2.to(torch.float32)
    	
    	f1 = self.fe(x1)

    	f = self.flatten(self.avg_pool(f1))
    	f = torch.cat((f, x2), dim=1)
    	return self.clf(f)

def predict(model, obs, obs_prev, action_prev, i):
    model = model.eval()
    # Get image from observation
    image, im1, right, left, overhead, wrist = get_image_from_observation(obs, obs_prev, i)
    
    # Save three first Images
    name_1 = "Imagenes GIF/Front/" + str(i) + ".png"
    name_2 = "Imagenes GIF/Right/" + str(i) + ".png"
    name_3 = "Imagenes GIF/Left/" + str(i) + ".png"
    name_4 = "Imagenes GIF/Overhead/" + str(i) + ".png"
    name_5 = "Imagenes GIF/Wrist/" + str(i) + ".png"

    im1.save(name_1, "PNG")
    right.save(name_2, "PNG")
    left.save(name_3, "PNG")
    overhead.save(name_4, "PNG")
    wrist.save(name_5, "PNG")
    
    transformations = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406, 0.485, 0.456, 0.406), (0.229, 0.224, 0.225, 0.229, 0.224, 0.225))])
    
    image = transformations(image).unsqueeze(dim=0)

    with torch.no_grad():
        pred = model((image, action_prev))
    pred = pred.detach().cpu().numpy()
    prediction = pred[0, :].tolist()
    prediction.append(1)
    #print("PRED", prediction, " FIN")
    return prediction
    
def get_image_from_observation(obs, obs_prev, i):
	im1 = Image.fromarray(obs.front_rgb)
	right = Image.fromarray(obs.right_shoulder_rgb)
	left = Image.fromarray(obs.left_shoulder_rgb)
	overhead = Image.fromarray(obs.overhead_rgb)
	wrist = Image.fromarray(obs.wrist_rgb)

	if ((i % episode_length == 0) or (i == 0)):
		im2 = im1
	else:
		im2 = Image.fromarray(obs_prev.front_rgb)
		
	image = np.concatenate((im1, im2), axis=2)

	return image, im1, right, left, overhead, wrist # Le pasamos la im1 para que la guarde y haga GIFs en predict()

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
headless=False)
env.launch()

task = env.get_task(ReachTarget)
print("TASK ", task)
print("TARGET ", ReachTarget)

# Load model
model = torch.load('model_pytorch_0178', map_location ='cpu')

#obs = env.reset()
episode_length = 100
obs = None
obs_prev = None
terminate = False
Done = False
i = 0
its = 0
action_prev = np.array([0,0,0,0,0,0,0], dtype="float32")
scaler = joblib.load("MinMax_scaler.save") 

while not terminate:
	if i % episode_length == 0:
		print('Reset Episode')
		descriptions, obs = task.reset()
		its = its + 1
		obs_prev = obs
	action = predict(model, obs, obs_prev, action_prev, i)
	action_no_norm = np.array(action, dtype="float32")
	
	if NORM == str(1): # En caso de tener que desnormalizar las acciones 
		print(action)
		action = np.array(action[:-1])
		action = action[np.newaxis, :]
		action = scaler.inverse_transform(action)
		print(action)
		action = np.append(action, 1)
	obs_prev = obs
	obs, reward, terminate = task.step(action)
	print('\nIteration: ' , i)
	print('Action: ' , action)
	i = i + 1
	#print("ACTION ", action)
	#print(type(action))
	#print("ACTION NO NORM ", action_no_norm)
	#print(type(action_no_norm))
	action_prev = action_no_norm[:-1]
	
	if its >= 5:
		terminate = True
		
print("Done!")
env.shutdown()
