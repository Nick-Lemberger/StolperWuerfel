import numpy as np
import random
import cube
import time

# Grid where the game world lives
world_size = 64
g1 = np.zeros((world_size, world_size), dtype=np.uint8)

# Panel pixel definition: [r,g,b] where each color is a 64x64 uint8 matrix 
panel_data = np.zeros((64, 64, 3), dtype=np.uint8) 

start_density = 0.005;


UDP_IP = '192.168.178.50'
UDP_PORT = 2000
c = cube.Cube(UDP_IP,UDP_PORT)

# Setup

for p in range(6):
    panel_data = panel_data * 0        
    panel_data[0,0,:] = [255,255,255]
    panel_data[1,0,:] = [255,255,255]
    panel_data[2,0,:] = [255,255,255]
    for i in range(p+1):
        panel_data[10+i,0,:] = [255,0,0]
    c.sendFullPanel(p,panel_data)
        
