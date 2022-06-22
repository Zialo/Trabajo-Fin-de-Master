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

    	x1, x2, x3, x4, x5, x6, x7, x8, x9 = x

    	x9 = torch.from_numpy(x9)
    	x9 = x9.unsqueeze(0)
    	x9 = x9.to(torch.float32)
    	
    	f1 = self.fe(x1)
    	f2 = self.fe(x2)
    	f3 = self.fe(x3)
    	f4 = self.fe(x4)
    	f5 = self.fe(x5)
    	f6 = self.fe(x6)
    	f7 = self.fe(x7)
    	f8 = self.fe(x8)
    	f = torch.cat((f1, f2, f3, f4, f5, f6, f7, f8), dim=1)
    	f = self.flatten(self.avg_pool(f))
    	f = torch.cat((f, x9), dim=1)
    	return self.clf(f)

def predict(model, obs, obs_prev, action_prev, i):
    model = model.eval()
    # Get image from observation
    im1, im2, im3, im4, im5, im6, im7, im8 = get_image_from_observation(obs, obs_prev, i)
    
    # Save three first Images
    name_1 = "Imagenes GIF/Front/" + str(i) + ".png"
    name_2 = "Imagenes GIF/Right/" + str(i) + ".png"
    name_3 = "Imagenes GIF/Left/" + str(i) + ".png"
    name_4 = "Imagenes GIF/Overhead/" + str(i) + ".png"

    im1.save(name_1, "PNG")
    im2.save(name_2, "PNG")
    im3.save(name_3, "PNG")
    im4.save(name_4, "PNG")
    
    transformations = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    
    im1 = transformations(im1).unsqueeze(dim=0)
    im2 = transformations(im2).unsqueeze(dim=0)
    im3 = transformations(im3).unsqueeze(dim=0)
    im4 = transformations(im4).unsqueeze(dim=0)
    im5 = transformations(im5).unsqueeze(dim=0)
    im6 = transformations(im6).unsqueeze(dim=0)
    im7 = transformations(im7).unsqueeze(dim=0)
    im8 = transformations(im8).unsqueeze(dim=0)

    with torch.no_grad():
        pred = model((im1, im2, im3, im4, im5, im6, im7, im8, action_prev))
    pred = pred.detach().cpu().numpy()
    prediction = pred[0, :].tolist()
    prediction.append(1)
    return prediction
    
def get_image_from_observation(obs, obs_prev, i):
	im1 = Image.fromarray(obs.front_rgb)
	im2 = Image.fromarray(obs.left_shoulder_rgb)
	im3 = Image.fromarray(obs.right_shoulder_rgb)
	im4 = Image.fromarray(obs.overhead_rgb)
	
	if ((i % episode_length == 0) or (i == 0)):
		im5 = im1
		im6 = im2
		im7 = im3
		im8 = im4

	else:
		im5 = Image.fromarray(obs_prev.front_rgb)
		im6 = Image.fromarray(obs_prev.left_shoulder_rgb)
		im7 = Image.fromarray(obs_prev.right_shoulder_rgb)
		im8 = Image.fromarray(obs_prev.overhead_rgb)
	
	return im1, im2, im3, im4, im5, im6, im7, im8

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
model = torch.load('model_pytorch_0124_notnorm', map_location ='cpu')

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
		print("Norm")
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
	if NORM != str(1):
		action_prev = np.asarray(action_prev)
	if its >= 5:
		terminate = True
		
print("Done!")
env.shutdown()
