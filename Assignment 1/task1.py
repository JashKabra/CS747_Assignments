"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the base Algorithm class that all algorithms should inherit
from. Here are the method details:
    - __init__(self, num_arms, horizon): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).
    
    - get_reward(self, arm_index, reward): This method is called just after the 
        give_pull method. The method should update the algorithm's internal
        state based on the arm that was pulled and the reward that was received.
        (The value of arm_index is the same as the one returned by give_pull.)

We have implemented the epsilon-greedy algorithm for you. You can use it as a
reference for implementing your own algorithms.
"""

import numpy as np
import math
# Hint: math.log is much faster than np.log for scalars

class Algorithm:
    def __init__(self, num_arms, horizon):
        np.random.seed(5)
        self.num_arms = num_arms
        self.horizon = horizon
        self.rewards = np.zeros(num_arms)
        self.counts = np.zeros(num_arms)
        self.timestep = 0 
    
    def give_pull(self):
        raise NotImplementedError
    
    def get_reward(self, arm_index, reward):
        raise NotImplementedError

# Example implementation of Epsilon Greedy algorithm
class Eps_Greedy(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # Extra member variables to keep track of the state
        self.eps = 0.1
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
    
    def give_pull(self):
        if np.random.random() < self.eps:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.values)
    
    def get_reward(self, arm_index, reward):
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value


# START EDITING HERE
# You can use this space to define any helper functions that you need
def KL_diverg(x,y):
    if(x==0):
        return -math.log(1-y)
    if(x==1):
        return -math.log(y)
    return x*math.log(x/y)+(1-x)*math.log((1-x)/(1-y))

def binary_search(low,high,target_val,p):
    if(p==1):
        return 1
    mid=(low+high)/2
    if(low<high):
        if(KL_diverg(p,mid)>target_val):
            high=mid-0.01
        else:
            low=mid+0.01
        return binary_search(low,high,target_val,p)
    else:
        return mid
# END EDITING HERE

class UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.ucb=np.zeros(num_arms)
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        self.timestep+=1
        if self.timestep <= self.num_arms:
            return self.timestep-1
        else:
            return np.argmax(self.ucb)
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.counts[arm_index]+=1
        self.rewards[arm_index]+=reward
        if self.timestep < self.num_arms:
            return None
        ln=math.log(self.timestep)
        time_arr=np.empty(self.num_arms)
        time_arr.fill(2*ln)
        #self.ucb=np.divide(self.rewards,self.counts)+np.sqrt(np.divide(time_arr,self.counts))
        self.ucb=self.rewards/self.counts+np.sqrt(time_arr/self.counts)
        # END EDITING HERE

class KL_UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.c=3
        self.ucb_kl=np.zeros(num_arms)
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        self.timestep+=1
        if self.timestep <= self.num_arms:
            return self.timestep-1
        else:
            return np.argmax(self.ucb_kl)
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.counts[arm_index]+=1
        self.rewards[arm_index]+=reward
        if self.timestep < self.num_arms:
            return None
        ln=math.log(self.timestep)
        lnln=math.log(ln)
        for i in range(self.num_arms):
            self.ucb_kl[i]=binary_search(self.rewards[i]/self.counts[i],0.99,(ln+self.c*lnln)/self.counts[i],self.rewards[i]/self.counts[i])
        # END EDITING HERE


class Thompson_Sampling(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.beta=np.zeros(num_arms)
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        self.timestep+=1
        self.beta=np.random.beta(self.rewards+1,self.counts-self.rewards+1)
        return np.argmax(self.beta)
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.counts[arm_index]+=1
        self.rewards[arm_index]+=reward
        # for i in range(self.num_arms):
        #     self.beta[i]=np.random.beta(self.rewards[i]+1,self.counts[i]-self.rewards[i]+1)
        # END EDITING HERE
