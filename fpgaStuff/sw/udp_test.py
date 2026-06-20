#!/bin/python3.7
import socket
import numpy as np
import math
import time

pi = math.pi

UDP_IP = '192.168.178.50'
UDP_PORT = 2000


num_rows = 64
num_cols = num_rows
output_size = num_rows
panel_num = 6

header_size = 2
data_size = 3
hbuf = np.zeros((1, header_size), dtype=np.uint8)
dbuf = np.zeros((num_rows, data_size), dtype=np.uint8)

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

r = np.uint8(30)
g = np.uint8(30)
b = np.uint8(30)
panel = np.uint8(0)
addr = np.uint8(0)

i = 0

while(1):
    #time.sleep(0.001)
    for p in range(panel_num):
        for y in range(int(num_rows)):
            for x in range(num_cols):
                #g = (255-x*4) * (1-y/63) 
                #b = x*4 * (1-y/63) 
                #r = y*4 * (1-x/63)
            
                phase = i/128*pi
                b = 255 * (0.5+math.cos(x/64*pi+phase)/2)**1.0
                r = 255 * (0.5+math.cos(x/64*pi+phase+2*pi/3)/2)**1.0
                g = 255 * (0.5+math.cos(x/64*pi+phase+2*pi*2/3)/2)**1.0
            
                dbuf[x] = [b,g,r]
            packet = bytes([p+1, y]) + dbuf.tobytes() # Header + data payload
            #print(packet)
            s.sendto(packet, (UDP_IP, UDP_PORT))
        i = i+1
exit()





