"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the AlgorithmManyArms class. Here are the method details:
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
"""

import numpy as np

# START EDITING HERE
# You can use this space to define any helper functions that you need
# END EDITING HERE

class AlgorithmManyArms:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        # Horizon is same as number of arms
        self.rewards = np.zeros(num_arms)
        self.counts = np.zeros(num_arms)
        self.beta = np.zeros(num_arms)
        self.max_mean = 1-1/num_arms
        self.empirical_mean = np.ones(num_arms)*0.5*self.max_mean
        np.random.seed(5)
    def give_pull(self):
        # START EDITING HERE
        return np.argmax(self.beta)
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.counts[arm_index]+=1
        self.rewards[arm_index]+=reward
        self.empirical_mean[arm_index]=self.rewards[arm_index]/self.counts[arm_index]
        if(np.max(self.empirical_mean)>=self.max_mean*0.97):
            self.beta=np.zeros(self.num_arms)
            self.beta[np.argmax(self.empirical_mean)]=1
        else:
            self.beta=np.random.beta(self.rewards+1,self.counts-self.rewards+1)
        # END EDITING HERE
