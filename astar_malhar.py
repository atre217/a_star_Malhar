import heapq
import numpy as np
import math
import cv2
import pygame
import time
import itertools
import threading
import sys
import matplotlib.pyplot as plt
#  Pre defined values
map_width=600
map_height=250
step_size=1


# Defining slope and constant for lines 
def lines(point_A, point_B):
    try:
        m = (point_B[1]-point_A[1])/(point_B[0] - point_A[0])
        c = (point_B[1] - m * point_B[0])
        return m, c
    except:
        return point_A[1], point_A[1]

# Defining the obstacle space
def obs_space(width_map,height_map):
    coords=[]
    coords_scaled=[]
    for i in range(width_map ): 
        for j in range(height_map ): 
            coords.append((i,j)) 

    # Scaling the obstacle space
    for i in range(width_map*2 ): 
        for j in range(height_map*2 ): 
            coords_scaled.append((round(i/2),round(j/2))) 
            
    obs= []
    scal_obs=[]
    
    # Clearance
    clearance = 5 +5 #Celarance+Radius
    # Points for Hexagon
    H1 = (235.05, 162.5)
    H2 = (300, 200)
    H3 = (364.95, 162.5) 
    H4 = (364.95, 87.5)
    H5 = (300, 50)
    H6 = (235.05, 87.5)  
    #lines 
    hm_12,hc_12 = lines(H1,H2)
    hm_23,hc_23 = lines(H2, H3) 
    hm_65, hc_65 = lines(H6,H5)
    hm_54, hc_54 = lines(H5, H4) 
    
    # Points for Triangle
    T1 = (460,225)
    T2 = (510,125)
    T3 = (460,25)
    # Lines
    tm_12,tc_12 = lines(T1,T2)
    tm_23,tc_23 = lines(T2,T3)
     
    # Plotting the lines
    for pts in coords_scaled:
        x, y = pts[0], pts[1] 
        
        if x<= clearance or y<=clearance or x>= width_map-clearance or  y>= height_map-clearance:
            scal_obs.append((x,y))
        if x>100 -clearance and x<150 +clearance and y >150-clearance and y <250:
            scal_obs.append((x,y))
        if x>100-clearance and x<150+clearance and y >=0 and y <100+clearance:
            scal_obs.append((x,y))
        if x > 235.05 -  clearance and x < 364.95 + clearance:
            if (y - hm_12*x < hc_12  + clearance) and  (y - hm_23*x < hc_23 + clearance) and  (y - hm_65*x > hc_65 - clearance) and  (y - hm_54*x > hc_54  - clearance)  :
                scal_obs.append((x,y))
        if x>460-clearance and x<510 + clearance:
            if  (y - tm_12*x < tc_12 + clearance) and (y - tm_23*x > tc_23 - clearance)  :
                scal_obs.append((x,y))
                
    for pts in coords:
        x, y = pts[0], pts[1] 
        
        if x<= clearance or y<=clearance or x>= width_map-clearance or  y>= height_map-clearance:
            obs.append((x,y))
        if x>100 -clearance and x<150 +clearance and y >150-clearance and y <250:
            obs.append((x,y))
        if x>100-clearance and x<150+clearance and y >=0 and y <100+clearance:
            obs.append((x,y))
        if x > 235.05 -  clearance and x < 364.95 + clearance:
            if (y - hm_12*x < hc_12  + clearance) and  (y - hm_23*x < hc_23 + clearance) and  (y - hm_65*x > hc_65 - clearance) and  (y - hm_54*x > hc_54  - clearance)  :
                obs.append((x,y))
        if x>460-clearance and x<510 + clearance:
            if  (y - tm_12*x < tc_12 + clearance) and (y - tm_23*x > tc_23 - clearance)  :
                obs.append((x,y))
    return scal_obs,obs

