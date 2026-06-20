#!/bin/python3.7
import socket
import numpy as np
import math

class Cube():
    num_rows = 64
    num_cols = num_rows
    output_size = num_rows
    panel_num = 6

    header_size = 2
    data_size = 3
    dbuf = np.zeros((num_rows, data_size), dtype=np.uint8)

    panel = np.uint8(0)
    addr = np.uint8(0)
    
    def __init__(self, UDP_IP, UDP_PORT):
        self.UDP_IP = UDP_IP
        self.UDP_PORT = UDP_PORT
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
    def sendPixel(self,panel,y,data):
        r = data[:,0]
        g = data[:,1]
        b = data[:,2]          
        self.dbuf[:] = np.transpose([b,g,r], (1,0))            
        packet = bytes([panel+1, y]) + self.dbuf.tobytes() # Header + data payload
        self.s.sendto(packet, (self.UDP_IP, self.UDP_PORT))   
    
    def sendRow(self,panel,y,data):
        r = data[:,0]
        g = data[:,1]
        b = data[:,2]          
        self.dbuf[:] = np.transpose([b,g,r], (1,0))            
        packet = bytes([panel+1, y]) + self.dbuf.tobytes() # Header + data payload
        self.s.sendto(packet, (self.UDP_IP, self.UDP_PORT))
    
    def sendFullPanel(self,panel,data):
        for y in range(self.num_rows):
            r = data[:,y,0]
            g = data[:,y,1]
            b = data[:,y,2]          
            self.dbuf[:] = np.transpose([b,g,r], (1,0))            
            packet = bytes([panel+1, y]) + self.dbuf.tobytes() # Header + data payload
            self.s.sendto(packet, (self.UDP_IP, self.UDP_PORT))






