import matplotlib.pyplot as plt 
import matplotlib.animation as animation 
import numpy as np 
import os
from PIL import Image
plt.style.use('dark_background')
import pandas as pd
import fnmatch

class MyAnimation:
    def __init__(self, data):
        self.fig = plt.figure()
        self.data = np.array(data)
        self.ax = plt.axes(xlim=(0, self.data.shape[0]), ylim=(self.data.min()-0.1, self.data.max()+0.1))
        self.lines = [self.ax.plot([], [], lw=2, label=f'pos{i}') for i in range(self.data.shape[1])]
        self.lines = list(map(lambda x: x[0], self.lines))
        plt.legend()
        self.xdata = [[] for i in range(self.data.shape[1])]
        self.ydata = [[] for i in range(self.data.shape[1])]
        
    def init_animation(self):
        # creating an empty plot/frame
        for i in range(len(self.lines)):
            self.lines[i].set_data([], [])
        return self.lines
    
    def animate(self, i): 
        # appending new points to x, y axes points list 
        for j in range(self.data.shape[1]):
            self.xdata[j].append(i)
            self.ydata[j].append(self.data[i, j])
            self.lines[j].set_data(self.xdata[j], self.ydata[j])
        return self.lines

# Actions
'''
sub_df = df.iloc[:49, :]
a = MyAnimation(sub_df)
anim = animation.FuncAnimation(a.fig, a.animate, init_func=a.init_animation, frames=sub_df.shape[0], interval=20, blit=True) 
# save the animation as mp4 video file 
anim.save('rlbench_actions.gif',writer='imagemagick') 
'''
aux=list(np.arange(len(os.listdir("Imagenes GIF/Front"))))
aux2=[]
for i in aux:
    aux2.append(str(aux[i])+".png")
    
inicio = 0   
final = 199 
# Images
for p in range(5):

	fig = plt.figure()
	name = "Imagenes GIF/Gif_" + str(p) + ".gif"
	all_imagesFront = [Image.open(os.path.join("Imagenes GIF/Front/", i)) for i in aux2[inicio:final]]
	all_imagesLeft = [Image.open(os.path.join("Imagenes GIF/Left/", i)) for i in aux2[inicio:final]]
	all_imagesRight = [Image.open(os.path.join("Imagenes GIF/Right/", i)) for i in aux2[inicio:final]]
	all_imagesOver = [Image.open(os.path.join("Imagenes GIF/Overhead/", i)) for i in aux2[inicio:final]]
	all_imagesWrist = [Image.open(os.path.join("Imagenes GIF/Wrist/", i)) for i in aux2[inicio:final]]
	
	inicio = final + 1
	final = final + 200

	tam=all_imagesFront[0].size
	im=Image.new("RGB",(tam[0]*5,tam[1]))

	plt_images = []
	for i in range(len(all_imagesFront)):
	    im.paste(all_imagesLeft[i],(0,0))
	    im.paste(all_imagesFront[i],(tam[0],0))
	    im.paste(all_imagesRight[i],(tam[0]*2,0))
	    im.paste(all_imagesOver[i],(tam[0]*3,0))
	    im.paste(all_imagesWrist[i],(tam[0]*4,0))
	    
	    im2=plt.imshow(im, animated=True)
	    
	    plt_images.append([im2])
	    
	for j in range(15):
	    plt_images.append([im2])
	aniConcatenada = animation.ArtistAnimation(fig, plt_images, interval=200, blit=True)
	aniConcatenada.save(name, writer='imagemagick') 
	print(name + " DONE")
