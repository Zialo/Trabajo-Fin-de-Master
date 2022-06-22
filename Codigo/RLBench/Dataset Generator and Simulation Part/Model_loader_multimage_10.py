import os
import sys
from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget
import numpy as np
import pandas as pd
import pickle
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

# PREDICT
class MultiImage(nn.Module):
    def __init__(self, fe, clf):
        super(MultiImage, self).__init__()
        self.fe = fe
        self.clf = clf
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
    def forward(self, x):
        x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11 = x
        f1 = self.fe(x1)
        f2 = self.fe(x2)
        f3 = self.fe(x3)
        f4 = self.fe(x4)
        f5 = self.fe(x5)
        f6 = self.fe(x6)
        f7 = self.fe(x7)
        f8 = self.fe(x8)
        f9 = self.fe(x9)
        f10 = self.fe(x10)
        f = torch.cat((f1, f2, f3, f4, f5, f6, f7, f8, f9, f10), dim=1)
        f = self.flatten(self.avg_pool(f))
        f = torch.cat((f, x11), dim=1)
        return self.clf(f)

def predict(model, obs, obs_prev, action_prev, i):
    model = model.eval()
    # Get image from observation
    im1, im2, im3, im4, im5, im6, im7, im8, im9, im10 = get_image_from_observation(obs, obs_prev, i)
    
    # Save three first Images
    name_1 = "Imagenes GIF/Front/" + str(i) + ".png"
    name_2 = "Imagenes GIF/Right/" + str(i) + ".png"
    name_3 = "Imagenes GIF/Left/" + str(i) + ".png"
    name_4 = "Imagenes GIF/Overhead/" + str(i) + ".png"
    name_5 = "Imagenes GIF/Depth/" + str(i) + ".png"
    im1.save(name_1, "PNG")
    im2.save(name_2, "PNG")
    im3.save(name_3, "PNG")
    im4.save(name_4, "PNG")
    im5.save(name_5, "PNG")
    
    transformations = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    
    im1 = transformations(im1).unsqueeze(dim=0)
    im2 = transformations(im2).unsqueeze(dim=0)
    im3 = transformations(im3).unsqueeze(dim=0)
    im4 = transformations(im4).unsqueeze(dim=0)
    im5 = transformations(im5).unsqueeze(dim=0)
    im6 = transformations(im6).unsqueeze(dim=0)
    im7 = transformations(im7).unsqueeze(dim=0)
    im8 = transformations(im8).unsqueeze(dim=0)
    im9 = transformations(im9).unsqueeze(dim=0)
    im10 = transformations(im10).unsqueeze(dim=0)
    with torch.no_grad():
        pred = model((im1, im2, im3, im4, im5, im6, im7, im8, im9, im10, action_prev))
    pred = pred.detach().cpu().numpy()
    prediction = pred[0, :].tolist()
    prediction.append(1)
    return prediction
    
def get_image_from_observation(obs, obs_prev, i):
	im1 = Image.fromarray(obs.front_rgb)
	im2 = Image.fromarray(obs.left_shoulder_rgb)
	im3 = Image.fromarray(obs.right_shoulder_rgb)
	im4 = Image.fromarray(obs.overhead_rgb)
	im5 = utils.float_array_to_rgb_image(obs.front_depth, scale_factor=DEPTH_SCALE)
	
	if ((i % episode_length == 0) or (i == 0)):
		im6 = im1
		im7 = im2
		im8 = im3
		im9 = im4
		im10 = im5
	else:
		im6 = Image.fromarray(obs_prev.front_rgb)
		im7 = Image.fromarray(obs_prev.left_shoulder_rgb)
		im8 = Image.fromarray(obs_prev.right_shoulder_rgb)
		im9 = Image.fromarray(obs_prev.overhead_rgb)
		im10 = utils.float_array_to_rgb_image(obs_prev.front_depth, scale_factor=DEPTH_SCALE)
	
	return im1, im2, im3, im4, im5, im6, im7, im8, im9, im10 

obs_config = ObservationConfig()
obs_config.set_all(True)
obs_config.right_shoulder_camera.image_size = img_size
obs_config.left_shoulder_camera.image_size = img_size
obs_config.overhead_camera.image_size = img_size
obs_config.wrist_camera.image_size = img_size
obs_config.front_camera.image_size = img_size

#Store depth as 0 - 1
obs_config.right_shoulder_camera.depth_in_meters = False
obs_config.left_shoulder_camera.depth_in_meters = False
obs_config.overhead_camera.depth_in_meters = False
obs_config.wrist_camera.depth_in_meters = False
obs_config.front_camera.depth_in_meters = False

#We want to save the masks as rgb encodings.
obs_config.left_shoulder_camera.masks_as_one_channel = False
obs_config.right_shoulder_camera.masks_as_one_channel = False
obs_config.overhead_camera.masks_as_one_channel = False
obs_config.wrist_camera.masks_as_one_channel = False
obs_config.front_camera.masks_as_one_channel = False

action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)

env = Environment(
    action_mode = action_mode,
    obs_config=obs_config,
headless=False)
env.launch()

task = env.get_task(ReachTarget)
print(task)
print(ReachTarget)

# Load model
model = torch.load('model_pytorch_10_images_norm', map_location ='cpu')

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
	action = predict(model, obs, obs_prev, action_prev, i)
	if NORM == str(1): # En caso de tener que desnormalizar las acciones 
		'''
		#print(action) # Accion Normalizada por el modelo que trabaja con datos normalizados
		data = pd.read_csv('Min_Max_Values.csv', header = None) # CSV de dim (2,7) con los minimos y maximos de las acciones del conjunto de datos de prueba
		desnormalization = np.float32(np.array(data))
		
		# La formula para desnormalizar es: x = x_norm * (x_max - x_min) + x_min 
		
		for i in range(7):
			action[i] = action[i] * (desnormalization[1,i] - desnormalization[0,i]) + desnormalization[0,i]
		'''

		action = np.array(action[:-1])
		action = action[np.newaxis, :]
		action = scaler.inverse_transform(action)
		action = np.append(action,1)
	obs_prev = obs
	obs, reward, terminate = task.step(action)
	#print('\nIteration: ' , i)
	#print('Action: ' , action)
	i = i + 1
	action_prev = action[:-1]
	if its >= 5:
		terminate = True
		
print("Done!")
env.shutdown()
