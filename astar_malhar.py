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
                    
################# BACKTRACKING #################
def animate_bktrking():
    for c in itertools.cycle([".", "..", "..."]):
        if is_computing_backtrack:
            break
        sys.stdout.write('\rBacktracking ' + c)
        sys.stdout.flush()
        time.sleep(0.1)


def backtracking(par_idx,goal,start):
    back_track = []
    present= start
    back_track.append(present)
    goal_achieved = False
    global is_computing_backtrack
    computing = threading.Thread(target=animate_bktrking)
    computing.start()

    while goal_achieved == False:
        for coord,parent_cost in par_idx.items():
            for cost,parent in parent_cost.items():
                if coord==present:
                    if parent not in back_track:
                        back_track.append(present)
                    present = parent
                    if parent == goal:
                        goal_achieved = True
                        time.sleep(1)
                        is_computing_backtrack = True
                        
                        break
    back_track.append(goal)
    return back_track

is_computing_astar_algo = False
is_computing_backtrack = False

################# Visualizing the map #################

def visualize_map(map_width,map_height,obstacle_scaled,obstacle_cord,list_closed,back_track_coord):
    obs_map = np.zeros((map_width*2+1,map_height*2+1,3),np.uint8) 
    obs_map[obstacle_scaled*2]=255
    obs_set = set(obstacle_scaled)
    
    # Using pygame here
    pygame.init()
    gameDisplay = pygame.display.set_mode((map_width*2,map_height*2))
    pygame.surfarray.make_surface(obs_map)
    pygame.display.set_caption('A-Star Algorithm')
    
    gameDisplay.fill((0,0,0))
    # Adding obstacles 
    for coords in obs_set:
        pygame.draw.rect(gameDisplay, (0,0,255), [coords[0]*2 ,abs(250-coords[1])*2,1,1])
        pygame.display.update()
    # Adding explored region/ visited nodes
    for coords in list_closed:
        pygame.time.wait(10)
        pygame.draw.rect(gameDisplay, (0,255,0), [coords[0]*2 ,abs(250-coords[1])*2,1,1])
        pygame.display.update()
    # Adding back track path
    for coords in back_track_coord:    
        pygame.time.wait(10)
        pygame.draw.rect(gameDisplay, (0,255,255), [coords[0]*2,abs(250-coords[1])*2,1,1])
        pygame.display.update()
  
    pygame.quit()

    obs_map_3d =  np.zeros((map_height+1,map_width+1,3),np.uint8)
    for x,y in obstacle_cord: 
        obs_map_3d[(y,x)]=[255,0,255] 
        
    for x,y,d in back_track_coord:
        obs_map_3d[(250-y,x)]=[0,0,255] 
    backtrack_map = cv2.resize(obs_map_3d,(map_width*3,map_height*3))
    cv2.imshow('Backtrack Path',backtrack_map) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

################# MAIN FUNCTION #################
obstacle_scaled,obstacle_cord =obs_space(600,250)

# Display the map workspace
x, y = [], []
for i in obstacle_scaled:
    x.append(i[0])
    y.append(i[1])
plt.scatter(x,y,s=0.1,c='black')
plt.axis([0,600,0,250])
plt.title('Obstacle Map')
plt.grid(which='both')
plt.show()

try:

    while True: 
        # getting inputs from the user
        s_xc = int(input("\nPlease enter the starting x coordinate of the robot : "))
        s_yc = int(input("Please enter the starting y coordinate of the robot : "))
        s_orienation =int(input("Please enter the starting orientation of the robot : "))

        g_xc = int(input("\nPlease enter the goal x coordinate of the robot  : "))
        g_yc = int(input("Please enter the goal y coordinate of the robot  : "))
        g_orientation =int(input("Please enter the desired final orientation of robot  : "))

        step_size = int(input("\nPlease enter the step size: "))
        radius = int(input("Please enter the radius of robot : "))
        clearance = int(input("Please enter the clearance of robot: "))
        
        # If inputs are incorrect,
        if(s_xc>=map_width or g_xc>=map_width or g_yc>=map_height or g_yc>=map_height or s_xc<0 or g_xc<0 or g_yc<0 or g_yc<0):
            print("Try again: Point out of bounds... 0<x<599 and 0<y<249: ")
            continue
        
        elif((s_xc,s_yc)  in obstacle_cord ) or ((g_xc,g_yc) in obstacle_cord ):
            print("Try again: The point is on the obstacle space: ")
            continue

        elif s_orienation not in (0,30,60,-30,-60) or g_orientation not in (0,30,60,-30,-60):
            print("Try again: Please enter any one of '30,60,-30,-60,0' degrees for Orientation: ")
            continue

        if radius+ clearance !=10:
            print("Try again: Enter clearance and radius value not greater than 10: ")
        
        if step_size< 1 and step_size>10:
            print("Try again: Enter step size value between 1 to 10: ")
            continue           

        else: 
            start = (s_xc,s_yc,s_orienation) 
            goal =  (g_xc,g_yc,g_orientation)
            break

    # keepinga athreshold value for x due to triangle  
    if s_xc <460 :
        start_time = time.time()
        par_idx,list_closed,goal_new,isGoal = astar_algo(start,goal,map_width,map_height,3,obstacle_scaled)
        # Print final time
        print("Time explored = %2.3f seconds " % (time.time() - start_time))
        if(isGoal):
            back_track_coord =backtracking(par_idx,start,goal_new)
            print("\nTime required to find Path: ",time.time() - start_time, "seconds" )
            visualize_map(map_width,map_height,obstacle_scaled,obstacle_cord,list_closed,back_track_coord)
        
        else:
            print("Bactracking Unsuccessful")
    else :
        print("Failed to reach goal")
    

    
except:
    print("You have entered an invalid output please Run the program again")
    
print("Program Executed ")
