import random
import math
import numpy as np
import cube
from scipy.ndimage import gaussian_filter

# Red tracers
#cIncr = [12, 5, 5]
#cDecr = [1, 2, 2]
#cLim =  [255, 200, 200]

# Green tracers
#cIncr = [0, 5, 0]
#cDecr = [5, 1, 5]
#cLim =  [200, 255, 120]

# Fireflies
cIncr = [20, 10, 10]
cDecr = [1, 4, 1]
cLim =  [255, 150, 120]
        
lifeTime = 1000
pList = []
maxSpeed = 0.1
vField = np.zeros((64, 64, 6, 2), dtype=float) 

world_size = 64
world_buf = np.zeros((world_size, world_size,6), dtype=np.uint8) # Buffer for integer particle positions 
cube_buf = np.zeros((world_size, world_size,6,3), dtype=np.uint8) 
UDP_IP = '192.168.178.50'
UDP_PORT = 2000
c = cube.Cube(UDP_IP,UDP_PORT)

class Particle:
    def __init__(self):
        self.pos = [0,0] # x,y
        self.face = 0 #face on cube
        self.maxSpeed = 0 # scalar speed
        self.v = [1,0] # velocity, normalized
        self.tLeft = 0 # time left to life

def add_saturate_uint8(dst, src):
    tmp = dst.astype(np.int16) + src.astype(np.int16)
    np.clip(tmp, 0, 255, out=tmp)
    dst[:] = tmp

def updateWorldBuffer():
    global world_buf
    world_buf *= 0
    for p in pList:
        x = int(p.pos[0])
        y = int(p.pos[1])
        face = p.face
        world_buf[x,y,face] += 1
        
def sendToCube():
    d = world_buf.astype(np.int16)
    # Accumulate colors at different speeds for cool visual effect
    dRed = d*cIncr[0]-cDecr[0]
    dGreen = d*cIncr[1]-cDecr[1]
    dBlue = d*cIncr[2]-cDecr[2]
    for i in range(6):
            delta = [dRed[:,:,i],dGreen[:,:,i],dBlue[:,:,i]]
            panel_buf = cube_buf[:,:,i,:]
            add_saturate_uint8(panel_buf, np.transpose(delta, (1,2,0)))
            # Clip individual colors channels
            panel_buf[:,:,0] = panel_buf[:,:,0].clip(0,cLim[0]) # Red
            panel_buf[:,:,1] = panel_buf[:,:,1].clip(0,cLim[1])# Green
            panel_buf[:,:,2] = panel_buf[:,:,2].clip(0,cLim[2]) # Blue
            cube_buf[:,:,i,:] = panel_buf
            c.sendFullPanel(i,panel_buf)
    
def wrapCoords(face,x, y, vx, vy):
    # Inside current face
    if 0 <= x < world_size and 0 <= y < world_size:
        return [face, x, y, vx, vy]
    # Ring faces
    if face in (0,1,2,3):
        # Left/right wrap around ring
        if x < 0:
            return wrapCoords((face+1) % 4, world_size-1, y, vx, vy)
        if x >= world_size:
            return wrapCoords((face-1) % 4, 0, y, vx, vy)
        # Top edge -> face 4
        if y < 0:
            if face == 0:
                return wrapCoords(4, world_size-1, world_size-1-x, vy, -vx)
            elif face == 1:
                return wrapCoords(4, x, world_size - 1, vx, vy)
            elif face == 2:
                return wrapCoords(4, 0, x, -vy, vx)
            else: # face == 3:
                return wrapCoords(4, world_size-1-x, 0, -vx, -vy)

        # Bottom edge -> face 5
        if y >= world_size:
            if face == 0:
                return wrapCoords(5, world_size-1, x, -vy, vx)
            elif face == 1:
                return wrapCoords(5, x, 0, vx, vy)
            elif face == 2:
                return wrapCoords(5, 0, world_size-1-x, vy, -vx)
            else: # face == 3:
                return wrapCoords(5, world_size-1-x, world_size-1, -vx, -vy)

    # Face 4, top
    if face == 4:
        if x >= world_size:
            return wrapCoords(0, world_size-1-y, 0, -vy, vx)
        if x < 0:
            return wrapCoords(2, y, 0, vy, -vx)
        if y >= world_size:
            return wrapCoords(1, x, 0, vx, vy)
        if y < 0:
            return wrapCoords(3, world_size-1-x, 0, -vx, -vy)

    # Face 5 (rotated 180°)
    if face == 5:
        if x >= world_size:
            return wrapCoords(0, y, world_size-1, vy, -vx)
        if x < 0:
            return wrapCoords(2, world_size-1-y, world_size-1, -vy, vx)
        if y >= world_size:
            return wrapCoords(3, world_size-1-x, world_size-1, -vx, -vy)
        if y < 0:
            return wrapCoords(1, x, world_size-1, vx, vy)   

def advanceTime(dt=1):
    global plist
    toRemove = []
    for i in range(len(pList)):
        if(pList[i].tLeft < 1):
            toRemove.append(i)
    for i in toRemove:      
        pList.pop(i)
        
    for p in pList:
        p.tLeft -= 1
        x = p.pos[0]
        y = p.pos[1]
        vx = p.v[0]
        vy = p.v[1]
        face = p.face
        # New position from old velocity direction
        x = x+vx*dt
        y = y+vy*dt      
        # Warp position around cube, rotate velocity accordingly
        [face,x,y,vx,vy] = wrapCoords(face,x,y,vx,vy)
        # Update position
        p.pos[0] = x
        p.pos[1] = y
        p.face = face     
        # New velocity direction from vector field
        vx = vx + vField[int(x),int(y),face,0]
        vy = vy + vField[int(x),int(y),face,1]       
        # Set to fixed speed
        p.v[0] = p.maxSpeed *vx/math.sqrt(vx**2 + vy**2)
        p.v[1] = p.maxSpeed *vy/math.sqrt(vx**2 + vy**2)
            
def randomLandscape():
    for f in range(6):    
        for i in range(64):
            for j in range(64):
                if f == 4: # Scatter in all directions on top
                    vField[i,j,f,0] = 0.01*(random.random()*2-1)
                    vField[i,j,f,1] = 0.01*(random.random()*2-1)
                elif f in (0,1,2,3): # Scatter downwards
                    vField[i,j,f,0] = 0.05*(random.random()*2-1)
                    vField[i,j,f,1] = 0.01#+0.05*random.random()
        sigma = 2
        vField[:,:,f,0] = gaussian_filter(vField[:,:,f,0], sigma=sigma)
        vField[:,:,f,1] = gaussian_filter(vField[:,:,f,1], sigma=sigma)           

def lanuchParticle():
    global plist
    p = Particle()
    angle = 2*math.pi*(2*random.random()-1)
    p.v = [maxSpeed*math.cos(angle), maxSpeed*math.sin(angle)]
    p.pos = [31, 31]
    p.face = 4   
    p.maxSpeed = maxSpeed
    p.tLeft = lifeTime
    pList.append(p)    

randomLandscape()
step = 1
while(1):
    lanuchParticle()
    lanuchParticle()
    advanceTime()
    updateWorldBuffer()   
    sendToCube()
    if (step%1000==0):
        randomLandscape()
    step += 1  
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
