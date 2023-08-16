from importlib.resources import path
from gym_driving.assets.car import *
from gym_driving.envs.environment import *
from gym_driving.envs.driving_env import *
from gym_driving.assets.terrain import *

import time
import pygame, sys
from pygame.locals import *
import random
import math
import argparse
import numpy as np

# Do NOT change these values
TIMESTEPS = 1000
FPS = 30
NUM_EPISODES = 10

class Task1():

    def __init__(self):
        """
        Can modify to include variables as required
        """

        super().__init__()

    def next_action(self, state):
        """
        Input: The current state
        Output: Action to be taken
        TO BE FILLED
        """

        # Replace with your implementation to determine actions to be taken
        action_steer = None
        action_acc = None

        x = state[0]
        y = state[1]
        theta = state[3]
        if(theta<0):
            theta = 360 + theta
        v = state[2]

        angle = math.atan2(-y, 350-x)*180/math.pi
        if(angle<0):
            angle = 360 + angle

        #print(angle,theta,v)

        if(np.abs(angle-theta)>10 and np.abs(angle-theta)<350):
            action_acc = 0
        else:
            action_acc = 4
        
        if(angle-theta>0):
            if(angle-theta>180):
                action_steer = 0
            else:
                action_steer = 2
        else:
            if(theta-angle>180):
                action_steer = 2
            else:
                action_steer = 0

        action = np.array([action_steer, action_acc])  

        return action

    def controller_task1(self, config_filepath=None, render_mode=False):
        """
        This is the main controller function. You can modify it as required except for the parts specifically not to be modified.
        Additionally, you can define helper functions within the class if needed for your logic.
        """
    
        ######### Do NOT modify these lines ##########
        pygame.init()
        fpsClock = pygame.time.Clock()

        if config_filepath is None:
            config_filepath = '../configs/config.json'

        simulator = DrivingEnv('T1', render_mode=render_mode, config_filepath=config_filepath)

        time.sleep(3)
        ##############################################

        # e is the number of the current episode, running it for 10 episodes
        for e in range(NUM_EPISODES):
        
            ######### Do NOT modify these lines ##########
            
            # To keep track of the number of timesteps per epoch
            cur_time = 0

            # To reset the simulator at the beginning of each episode
            state = simulator._reset()
            
            # Variable representing if you have reached the road
            road_status = False
            ##############################################

            # The following code is a basic example of the usage of the simulator
            for t in range(TIMESTEPS):
        
                # Checks for quit
                if render_mode:
                    for event in pygame.event.get():
                        if event.type == QUIT:
                            pygame.quit()
                            sys.exit()

                action = self.next_action(state)
                state, reward, terminate, reached_road, info_dict = simulator._step(action)
                fpsClock.tick(FPS)

                cur_time += 1

                if terminate:
                    road_status = reached_road
                    break

            # Writing the output at each episode to STDOUT
            print(str(road_status) + ' ' + str(cur_time))

class Task2():

    def __init__(self):
        """
        Can modify to include variables as required
        """

        self.prev_angle=0

        super().__init__()

    def next_action(self, state):
        """
        Input: The current state
        Output: Action to be taken
        TO BE FILLED

        You can modify the function to take in extra arguments and return extra quantities apart from the ones specified if required
        """

        # Replace with your implementation to determine actions to be taken
        action_steer = None
        action_acc = None

        x = state[0]
        y = state[1]
        theta = state[3]
        v=state[2]

        can_go_vertical = True
        if(y>0):
            if(x>self.centres[0,0]-80 and x<self.centres[0,0]+80 and y>self.centres[0,1]+80):
                can_go_vertical = False
            if(x>self.centres[2,0]-80 and x<self.centres[2,0]+80 and y>self.centres[2,1]+80):
                can_go_vertical = False
        else:
            if(x>self.centres[1,0]-80 and x<self.centres[1,0]+80 and y<self.centres[1,1]-80):
                can_go_vertical = False
            if(x>self.centres[3,0]-80 and x<self.centres[3,0]+80 and y<self.centres[3,1]-80):
                can_go_vertical = False
        


        if(theta<0):
            theta = 360 + theta

        angle=0
        if(y>40):
            angle = 270
        elif(y<-40):
            angle = 90

        if(np.abs(y)<40 and np.abs(y)>20):
            angle = self.prev_angle

        if(not can_go_vertical):
            angle = 0

        self.prev_angle = angle

        #print(y,theta,angle)


        if(np.abs(angle-theta)>10 and np.abs(angle-theta)<350):
            action_acc = 0
        else:
            action_acc = 4
        
        if(angle-theta>0):
            if(angle-theta>180):
                action_steer = 0
            else:
                action_steer = 2
        else:
            if(theta-angle>180):
                action_steer = 2
            else:
                action_steer = 0

        action = np.array([action_steer, action_acc])  

        return action

    def controller_task2(self, config_filepath=None, render_mode=False):
        """
        This is the main controller function. You can modify it as required except for the parts specifically not to be modified.
        Additionally, you can define helper functions within the class if needed for your logic.
        """
        
        ################ Do NOT modify these lines ################
        pygame.init()
        fpsClock = pygame.time.Clock()

        if config_filepath is None:
            config_filepath = '../configs/config.json'

        time.sleep(3)
        ###########################################################

        # e is the number of the current episode, running it for 10 episodes
        for e in range(NUM_EPISODES):

            ################ Setting up the environment, do NOT modify these lines ################
            # To randomly initialize centers of the traps within a determined range
            ran_cen_1x = random.randint(120, 230)
            ran_cen_1y = random.randint(120, 230)
            ran_cen_1 = [ran_cen_1x, ran_cen_1y]

            ran_cen_2x = random.randint(120, 230)
            ran_cen_2y = random.randint(-230, -120)
            ran_cen_2 = [ran_cen_2x, ran_cen_2y]

            ran_cen_3x = random.randint(-230, -120)
            ran_cen_3y = random.randint(120, 230)
            ran_cen_3 = [ran_cen_3x, ran_cen_3y]

            ran_cen_4x = random.randint(-230, -120)
            ran_cen_4y = random.randint(-230, -120)
            ran_cen_4 = [ran_cen_4x, ran_cen_4y]

            ran_cen_list = [ran_cen_1, ran_cen_2, ran_cen_3, ran_cen_4]            
            eligible_list = []

            # To randomly initialize the car within a determined range
            for x in range(-300, 300):
                for y in range(-300, 300):

                    if x >= (ran_cen_1x - 110) and x <= (ran_cen_1x + 110) and y >= (ran_cen_1y - 110) and y <= (ran_cen_1y + 110):
                        continue

                    if x >= (ran_cen_2x - 110) and x <= (ran_cen_2x + 110) and y >= (ran_cen_2y - 110) and y <= (ran_cen_2y + 110):
                        continue

                    if x >= (ran_cen_3x - 110) and x <= (ran_cen_3x + 110) and y >= (ran_cen_3y - 110) and y <= (ran_cen_3y + 110):
                        continue

                    if x >= (ran_cen_4x - 110) and x <= (ran_cen_4x + 110) and y >= (ran_cen_4y - 110) and y <= (ran_cen_4y + 110):
                        continue

                    eligible_list.append((x,y))

            simulator = DrivingEnv('T2', eligible_list, render_mode=render_mode, config_filepath=config_filepath, ran_cen_list=ran_cen_list)
        
            # To keep track of the number of timesteps per episode
            cur_time = 0

            # To reset the simulator at the beginning of each episode
            state = simulator._reset(eligible_list=eligible_list)
            ###########################################################

            # The following code is a basic example of the usage of the simulator

            self.centres=np.array(ran_cen_list)
            road_status = False

            for t in range(TIMESTEPS):
        
                # Checks for quit
                if render_mode:
                    for event in pygame.event.get():
                        if event.type == QUIT:
                            pygame.quit()
                            sys.exit()

                action = self.next_action(state)
                state, reward, terminate, reached_road, info_dict = simulator._step(action)
                fpsClock.tick(FPS)

                cur_time += 1

                if terminate:
                    road_status = reached_road
                    break

            print(str(road_status) + ' ' + str(cur_time))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="config filepath", default=None)
    parser.add_argument("-t", "--task", help="task number", choices=['T1', 'T2'])
    parser.add_argument("-r", "--random_seed", help="random seed", type=int, default=0)
    parser.add_argument("-m", "--render_mode", action='store_true')
    parser.add_argument("-f", "--frames_per_sec", help="fps", type=int, default=30) # Keep this as the default while running your simulation to visualize results
    args = parser.parse_args()

    config_filepath = args.config
    task = args.task
    random_seed = args.random_seed
    render_mode = args.render_mode
    fps = args.frames_per_sec

    FPS = fps

    random.seed(random_seed)
    np.random.seed(random_seed)

    if task == 'T1':
        
        agent = Task1()
        agent.controller_task1(config_filepath=config_filepath, render_mode=render_mode)

    else:

        agent = Task2()
        agent.controller_task2(config_filepath=config_filepath, render_mode=render_mode)


###################################################################
# SARSA ATTEMPT 
######################################################################

#from importlib.resources import path
# from gym_driving.assets.car import *
# from gym_driving.envs.environment import *
# from gym_driving.envs.driving_env import *
# from gym_driving.assets.terrain import *

# import time
# import pygame, sys
# from pygame.locals import *
# import random
# import math
# import argparse

# # Do NOT change these values
# TIMESTEPS = 1000
# FPS = 30
# NUM_EPISODES = 10
# TRAIN_EPISODES = 1000

# class Task1():

#     def __init__(self):
#         """
#         Can modify to include variables as required
#         """
#         self.num_tiles = 10
#         self.num_x = 70
#         self.num_y = 70
#         self.num_theta = 12
#         self.num_v = 5
#         self.num_bins =[self.num_x, self.num_y, self.num_theta, self.num_v]
#         self.num_actions = 15
#         self.num_features = 4
#         self.weights = np.random.rand(self.num_x+1, self.num_y+1, self.num_theta+1, self.num_v+1, self.num_tiles,self.num_actions)
#         self.alpha = 0.2
#         self.gamma = 0.9
#         self.epsilon = 0.4
#         self.high = [350,350,180,13]
#         self.low = [-350,-350,-180,0]
#         super().__init__()

    
#     def tiles(self):
#         tiles = []
#         for i in range(self.num_features):
#             curr = []
#             tile_width = (self.high[i] - self.low[i])/self.num_bins[i]
#             tile_offset= tile_width/self.num_tiles
#             for j in range(self.num_actions):
#                 curr.append(np.linspace(self.low[i], self.high[i], self.num_bins[i]) + j*tile_offset)
#             tiles.append(curr)
#         self.tilings = np.array(tiles, dtype=object)

#     def get_tile_indices(self, state):
#         tile_indices = np.zeros((self.num_tiles, self.num_features))
#         for i in range(self.num_tiles):
#             for j in range(self.num_features):
#                 tile_indices[i,j] = np.digitize(state[j], self.tilings[j,i])
#         return tile_indices.astype(int)
                

#     def next_action(self, state):
#         """
#         Input: The current state
#         Output: Action to be taken
#         TO BE FILLED
#         """

#         # Replace with your implementation to determine actions to be taken
#         action_steer = None
#         action_acc = None

#         indices = self.get_tile_indices(state)
#         q_values = np.zeros((self.num_tiles, self.num_actions))
#         for i in range(self.num_tiles):
#             q_values[i,:] = self.weights[indices[i,0], indices[i,1], indices[i,2], indices[i,3], i, :]
#         q_values = np.sum(q_values, axis=0)
#         if np.random.rand() < self.epsilon:
#             action = np.random.randint(self.num_actions)
#         else:
#             action = np.argmax(q_values)
#         action_steer = action%3
#         action_acc = action//3
#         return [action_steer, action_acc]

#     def update_weights(self, state, action, reward, next_state, next_action):
#         indices = self.get_tile_indices(state)
#         next_indices = self.get_tile_indices(next_state)
#         for i in range(self.num_tiles):
#             target= reward + self.gamma * self.weights[next_indices[i,0], next_indices[i,1], next_indices[i,2], next_indices[i,3], i, next_action]
#             self.weights[indices[i,0], indices[i,1], indices[i,2], indices[i,3], i, action] += self.alpha * (target - self.weights[indices[i,0], indices[i,1], indices[i,2], indices[i,3], i, action])

#     def train(self,config_filepath=None):
#         ######### Do NOT modify these lines ##########
#         pygame.init()
#         fpsClock = pygame.time.Clock()

#         if config_filepath is None:
#             config_filepath = '../configs/config.json'

#         simulator = DrivingEnv('T1',render_mode=False,config_filepath=config_filepath)

#         time.sleep(3)
#         ##############################################

#         self.tiles()

#         # e is the number of the current episode, running it for 10 episodes
#         for e in range(TRAIN_EPISODES):
        
#             ######### Do NOT modify these lines ##########
            
#             # To keep track of the number of timesteps per epoch
#             cur_time = 0

#             # To reset the simulator at the beginning of each episode
#             state = simulator._reset()
            
#             # Variable representing if you have reached the road
#             road_status = False
#             ##############################################


#             # The following code is a basic example of the usage of the simulator
#             for t in range(TIMESTEPS):

#                 curr_action = self.next_action(state)
#                 next_state, reward, terminate, reached_road, info_dict = simulator._step(curr_action)
#                 next_action = self.next_action(next_state)
#                 self.update_weights(state, curr_action[0]*3 + curr_action[1], reward, next_state, next_action[0]*3 + next_action[1])
#                 fpsClock.tick(500)

#                 cur_time += 1

#                 if terminate:
#                     road_status = reached_road
#                     break

#             # Writing the output at each episode to STDOUT
#             print(str(road_status) + ' ' + str(cur_time))

#     def controller_task1(self, config_filepath=None, render_mode=False):
#         """
#         This is the main controller function. You can modify it as required except for the parts specifically not to be modified.
#         Additionally, you can define helper functions within the class if needed for your logic.
#         """
    
#         ######### Do NOT modify these lines ##########
#         pygame.init()
#         fpsClock = pygame.time.Clock()

#         if config_filepath is None:
#             config_filepath = '../configs/config.json'

#         simulator = DrivingEnv('T1', render_mode=render_mode, config_filepath=config_filepath)

#         time.sleep(3)
#         ##############################################

#         self.tiles()

#         # e is the number of the current episode, running it for 10 episodes
#         for e in range(NUM_EPISODES):
        
#             ######### Do NOT modify these lines ##########
            
#             # To keep track of the number of timesteps per epoch
#             cur_time = 0

#             # To reset the simulator at the beginning of each episode
#             state = simulator._reset()
            
#             # Variable representing if you have reached the road
#             road_status = False
#             ##############################################


#             # The following code is a basic example of the usage of the simulator
#             for t in range(TIMESTEPS):
        
#                 # Checks for quit
#                 if render_mode:
#                     for event in pygame.event.get():
#                         if event.type == QUIT:
#                             pygame.quit()
#                             sys.exit()

#                 curr_action = self.next_action(state)
#                 next_state, reward, terminate, reached_road, info_dict = simulator._step(curr_action)
#                 next_action = self.next_action(next_state)
#                 self.update_weights(state, curr_action[0]*3 + curr_action[1], reward, next_state, next_action[0]*3 + next_action[1])
#                 fpsClock.tick(FPS)

#                 cur_time += 1

#                 if terminate:
#                     road_status = reached_road
#                     break

#             # Writing the output at each episode to STDOUT
#             print(str(road_status) + ' ' + str(cur_time))

# class Task2():

#     def __init__(self):
#         """
#         Can modify to include variables as required
#         """

#         super().__init__()

#     def next_action(self, state):
#         """
#         Input: The current state
#         Output: Action to be taken
#         TO BE FILLED

#         You can modify the function to take in extra arguments and return extra quantities apart from the ones specified if required
#         """

#         # Replace with your implementation to determine actions to be taken
#         action_steer = None
#         action_acc = None

#         action = np.array([action_steer, action_acc])  

#         return action

#     def controller_task2(self, config_filepath=None, render_mode=False):
#         """
#         This is the main controller function. You can modify it as required except for the parts specifically not to be modified.
#         Additionally, you can define helper functions within the class if needed for your logic.
#         """
        
#         ################ Do NOT modify these lines ################
#         pygame.init()
#         fpsClock = pygame.time.Clock()

#         if config_filepath is None:
#             config_filepath = '../configs/config.json'

#         time.sleep(3)
#         ###########################################################

#         # e is the number of the current episode, running it for 10 episodes
#         for e in range(NUM_EPISODES):

#             ################ Setting up the environment, do NOT modify these lines ################
#             # To randomly initialize centers of the traps within a determined range
#             ran_cen_1x = random.randint(120, 230)
#             ran_cen_1y = random.randint(120, 230)
#             ran_cen_1 = [ran_cen_1x, ran_cen_1y]

#             ran_cen_2x = random.randint(120, 230)
#             ran_cen_2y = random.randint(-230, -120)
#             ran_cen_2 = [ran_cen_2x, ran_cen_2y]

#             ran_cen_3x = random.randint(-230, -120)
#             ran_cen_3y = random.randint(120, 230)
#             ran_cen_3 = [ran_cen_3x, ran_cen_3y]

#             ran_cen_4x = random.randint(-230, -120)
#             ran_cen_4y = random.randint(-230, -120)
#             ran_cen_4 = [ran_cen_4x, ran_cen_4y]

#             ran_cen_list = [ran_cen_1, ran_cen_2, ran_cen_3, ran_cen_4]            
#             eligible_list = []

#             # To randomly initialize the car within a determined range
#             for x in range(-300, 300):
#                 for y in range(-300, 300):

#                     if x >= (ran_cen_1x - 110) and x <= (ran_cen_1x + 110) and y >= (ran_cen_1y - 110) and y <= (ran_cen_1y + 110):
#                         continue

#                     if x >= (ran_cen_2x - 110) and x <= (ran_cen_2x + 110) and y >= (ran_cen_2y - 110) and y <= (ran_cen_2y + 110):
#                         continue

#                     if x >= (ran_cen_3x - 110) and x <= (ran_cen_3x + 110) and y >= (ran_cen_3y - 110) and y <= (ran_cen_3y + 110):
#                         continue

#                     if x >= (ran_cen_4x - 110) and x <= (ran_cen_4x + 110) and y >= (ran_cen_4y - 110) and y <= (ran_cen_4y + 110):
#                         continue

#                     eligible_list.append((x,y))

#             simulator = DrivingEnv('T2', eligible_list, render_mode=render_mode, config_filepath=config_filepath, ran_cen_list=ran_cen_list)
        
#             # To keep track of the number of timesteps per episode
#             cur_time = 0

#             # To reset the simulator at the beginning of each episode
#             state = simulator._reset(eligible_list=eligible_list)
#             ###########################################################

#             # The following code is a basic example of the usage of the simulator
#             road_status = False

#             for t in range(TIMESTEPS):
        
#                 # Checks for quit
#                 if render_mode:
#                     for event in pygame.event.get():
#                         if event.type == QUIT:
#                             pygame.quit()
#                             sys.exit()

#                 action = self.next_action(state)
#                 state, reward, terminate, reached_road, info_dict = simulator._step(action)
#                 fpsClock.tick(FPS)

#                 cur_time += 1

#                 if terminate:
#                     road_status = reached_road
#                     break

#             print(str(road_status) + ' ' + str(cur_time))

# if __name__ == '__main__':

#     parser = argparse.ArgumentParser()
#     parser.add_argument("-c", "--config", help="config filepath", default=None)
#     parser.add_argument("-t", "--task", help="task number", choices=['T1', 'T2'])
#     parser.add_argument("-r", "--random_seed", help="random seed", type=int, default=0)
#     parser.add_argument("-m", "--render_mode", action='store_true')
#     parser.add_argument("-f", "--frames_per_sec", help="fps", type=int, default=30) # Keep this as the default while running your simulation to visualize results
#     args = parser.parse_args()

#     config_filepath = args.config
#     task = args.task
#     random_seed = args.random_seed
#     render_mode = args.render_mode
#     fps = args.frames_per_sec

#     FPS = fps

#     random.seed(random_seed)
#     np.random.seed(random_seed)

#     if task == 'T1':
        
#         agent = Task1()
#         agent.train(config_filepath=config_filepath)
#         agent.controller_task1(config_filepath=config_filepath, render_mode=render_mode)

#     else:

#         agent = Task2()
#         agent.controller_task2(config_filepath=config_filepath, render_mode=render_mode)
