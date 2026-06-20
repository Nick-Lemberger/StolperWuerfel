import numpy as np
import random
import cube
import time

# Game of life rules.
# First list: survie rules
# Second list: birth rules

# Original rules
ruleS = [3,2,7]
ruleB = [3]

# The Blobb
#ruleS = [3,5]
#ruleB = [3,4]

# Grids where the game world lives
world_size = 64
old = np.zeros((world_size, world_size, 6), dtype=np.uint8)
new = np.zeros((world_size, world_size, 6), dtype=np.uint8)
start_density = 0.20;
cutoff_pop = 5000;

# How the colors change
# Preset: "Slow Winter"
cIncr = [4, 6, 10]
cDecr = [1, 1, 1]
cLim = [255, 255, 255]

# Panel pixel definition: [r,g,b] where each color is a 64x64 uint8 matrix 
cube_buf = np.zeros((64, 64,3,6), dtype=np.uint8) 
UDP_IP = '192.168.178.50'
UDP_PORT = 2000
c = cube.Cube(UDP_IP,UDP_PORT)

def getCell(face, x, y):
    # Inside current face
    if 0 <= x < world_size and 0 <= y < world_size:
        return old[x, y, face]
    # Ring faces
    if face in (0,1,2,3):
        # Left/right wrap around ring
        if x < 0:
            return getCell((face + 1) % 4, world_size-1, y)
        if x >= world_size:
            return getCell((face - 1) % 4, 0, y)
        # Top edge -> face 4
        if y < 0:
            if face == 0:
                return getCell(4, world_size-1, world_size-1-x)
            elif face == 1:
                return getCell(4, x, world_size - 1)
            elif face == 2:
                return getCell(4, 0, x)
            else: # face == 3:
                return getCell(4, world_size-1-x, 0)

        # Bottom edge -> face 5
        if y >= world_size:
            if face == 0:
                return getCell(5, world_size-1, x)
            elif face == 1:
                return getCell(5, x, 0)
            elif face == 2:
                return getCell(5, 0, world_size-1-x)
            else: # face == 3:
                return getCell(5, world_size-1-x, world_size-1)

    # Face 4, top
    if face == 4:
        if x >= world_size:
            return getCell(0, world_size-1-y, 0)
        if x < 0:
            return getCell(2, y, 0)
        if y >= world_size:
            return getCell(1, x, 0)
        if y < 0:
            return getCell(3, world_size-1-x, 0)

    # Face 5 (rotated 180°)
    if face == 5:
        if x >= world_size:
            return getCell(0, y, world_size-1)
        if x < 0:
            return getCell(2, world_size-1-y, world_size-1)
        if y >= world_size:
            return getCell(3, world_size-1-x, world_size-1)
        if y < 0:
            return getCell(1, x, world_size-1)        
        
def getNeighbors(x, y, face):
    total = 0
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            total += getCell(face, x + dx, y + dy)
    return total
    
def add_saturate_uint8(dst, src):
    tmp = dst.astype(np.int16) + src.astype(np.int16)
    np.clip(tmp, 0, 255, out=tmp)
    dst[:] = tmp
    
def max_colors(dst, src, vMax):
    tmp 
    np.maximum(tmp[0], vMax[0])
    np.maximum(tmp[1], vMax[0])
    np.maximum(tmp[2], vMax[0])
    
def sendToCube(d):
    d = d.astype(np.int16)
    # Accumulate colors at different speeds for cool visual effect
    dRed = d*cIncr[0]-cDecr[0]
    dGreen = d*cIncr[1]-cDecr[1]
    dBlue = d*cIncr[2]-cDecr[2]
    for i in range(6):
            delta = [dRed[:,:,i],dGreen[:,:,i],dBlue[:,:,i]]
            panel_buf = cube_buf[:,:,:,i]
            add_saturate_uint8(panel_buf, np.transpose(delta, (1,2,0)))
            # Clip individual colors channels
            panel_buf[:,:,0] = panel_buf[:,:,0].clip(0,cLim[0]) # Red
            panel_buf[:,:,1] = panel_buf[:,:,1].clip(0,cLim[1])# Green
            panel_buf[:,:,2] = panel_buf[:,:,2].clip(0,cLim[2]) # Blue
            cube_buf[:,:,:,i] = panel_buf
            c.sendFullPanel(i,panel_buf)

def randomColors():
    global cIncr, cDecr, cLim
    cIncr = [random.randint(2, 50), random.randint(2, 50), random.randint(2, 50)]
    cDecr = [random.randint(1, 5), random.randint(1, 5), random.randint(1, 5)]
    cLim = [random.randint(100, 255), random.randint(100, 255), random.randint(100, 255)]

# Setup, random start population
def seed_world():
    for i in range(6):
        for x in range(64):
            for y in range(64):
                new[x,y,i] = 0 if random.random() > start_density else 1
# Start
seed_world()
randomColors()
print(cIncr)
print(cDecr)
print(cLim)

while(1):
    # Clear and re-seed if overpopulated
    if(np.sum(old) > cutoff_pop):
        old *= 0
        new *= 0
        seed_world()
        randomColors()
        
    old = new.copy()
    sendToCube(old)
        
    # Calculate new world
    for i in range(6):
        for x in range(world_size):
            for y in range(world_size):
                c0 = old[x,y,i] # current cell
                env = getNeighbors(x,y,i) 
                if c0 == 1:
                    new[x,y,i] = 1 if env in ruleS else 0
                else:
                    new[x,y,i] = 1 if env in ruleB else 0 
                    
        
        
        

        
            
            
        
        
        

            
