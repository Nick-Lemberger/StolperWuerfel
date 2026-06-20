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

while(1):
    for p in range(6):
        for x in range(1,world_size-1):
            for y in range(1,world_size-1):
                g1[x,y] = 0 if random.random() > start_density else 1          
        panel_data = panel_data + 255
        panel_data = panel_data  * np.transpose([g1,g1,g1],(1,2,0))
        c.sendToPanel(p,panel_data)
    time.sleep(0.01)
        
