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

# Defining a function that can work with angles
def action_set(step_size,coord, orntn,map_width,map_height):
    x = round((step_size)*np.cos(np.deg2rad(orntn)) + coord[0],2)
    y = round((step_size)*np.sin(np.deg2rad(orntn)) + coord[1],2)

    if x>=0 and x<=map_width and y>=0 and y<= map_height:
        return((x,y),True)
    else:
        return(coord,False)

def map(coord,orntn,map_width,map_height,step_size,obs_list): 
    
    obs_set = set(obs_list)
    cost = {}

    # orientation of robot for negative 60 i.e. 300
    ang_neg_60,is_ang_neg_60 = action_set(step_size,coord, orntn - 60,map_width,map_height)
    if is_ang_neg_60 and (ang_neg_60[0],ang_neg_60[1]) not in obs_set:
        cost[(ang_neg_60[0],ang_neg_60[1],orntn-60)] = step_size
     # orientation of robot for negative 30 i.e. 330
    ang_neg_30,is_ang_neg_30 = action_set(step_size,coord, orntn - 30,map_width,map_height)#-30
    if is_ang_neg_30 and (ang_neg_30[0],ang_neg_30[1]) not in obs_set:
        cost[(ang_neg_30[0],ang_neg_30[1],orntn-30)] = step_size
     # orientation of robot for zero degrees
    ang_zero,is_ang_zero = action_set(step_size,coord, orntn + 0,map_width,map_height)
    if is_ang_zero and (ang_zero[0],ang_zero[1]) not in obs_set:
        cost[(ang_zero[0],ang_zero[1],orntn)] = step_size
     # orientation of robot for positive 30 
    ang_30,is_ang_30 = action_set(step_size,coord, orntn + 30,map_width,map_height)#30
    if is_ang_30 and (ang_30[0],ang_30[1]) not in obs_set: 
        cost[(ang_30[0],ang_30[1],orntn+30)] = step_size
     # orientation of robot positive for 60
    ang_60,is_ang_60 = action_set(step_size,coord, orntn + 60,map_width,map_height)#60
    if is_ang_60 and (ang_60[0],ang_60[1]) not in obs_set:
        cost[(ang_60[0],ang_60[1],orntn+60)] = step_size
        
    return cost       

################# ASTAR ALGORITHM #################
def animate_astar_algo():
    for c in itertools.cycle([".", "..", "..."]):
        if is_computing_astar_algo:
            break
        sys.stdout.write('\rRunning the A-star Algorithm, Please wait' + c)
        sys.stdout.flush()
        time.sleep(0.1)


def astar_algo(start,goal,map_width,map_height,step_size,obs_list):
    
    list_cost = {}
    list_closed = []

    #Contains the parent node and the cost taken to reach the present node
    par_idx = {}
    print("Starting Co-ordinates :",start)
    print("Goal Co-ordinates :",goal)
    print("Step Size : ", step_size)
    list_cost[start]=0
    list_open = [(0,start)]
    goal_achieved = False
    count=0

    computing = threading.Thread(target=animate_astar_algo)
    computing.start()
    global is_computing_astar_algo

    obs_set = set(obs_list)
    while len(list_open)>0 and goal_achieved == False:
        count = count+1
        totalC, cord_parent = heapq.heappop(list_open) 
        posn_parent = (cord_parent[0],cord_parent[1])
        orntn = cord_parent[2]
        
        neighbours = map(posn_parent,orntn,map_width,map_height,step_size,obstacle_scaled)
        if posn_parent not in obs_set:
            for key, cost in neighbours.items():
                list_cost[key]=math.inf
                
            for coord, cost in neighbours.items():
                if(coord not in  list_closed) and (coord not in obs_list):
                    coord_round = (round(coord[0]),round(coord[1]),coord[2])
                    list_closed.append(coord_round)
                    Cost2Come = cost 
                    Cost2Go = math. dist((coord[0],coord[1]),(goal[0],goal[1]))  # h(n)
                    TotalCost = Cost2Come + Cost2Go   # f(n)
                    
                    if TotalCost < list_cost[coord] or coord not in list_open :
                        
                        par_idx[coord_round]={}
                        par_idx[coord_round][TotalCost] = cord_parent
                        list_cost[coord_round]=TotalCost
                        heapq.heappush(list_open, (TotalCost, coord_round))

                    # Step size determines the threshold
                    if ((coord_round[0]-goal[0])**2 + (coord_round[1]-goal[1])**2 <= (step_size)**2) and coord_round[2]==goal[2] :
                        print("\nFinal Node :",coord_round)
                        print('GOAL  Reached !!')
                        print("Total Cost :  ",TotalCost)
                        goal_achieved = True
                        time.sleep(0)
                        is_computing_astar_algo = True
                        return par_idx,list_closed,coord_round,True

                    
    return par_idx,list_closed,False
                    

