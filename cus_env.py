import gym
import numpy as np
import random
class TicTacToeEnv(gym.Env):
    def __init__(self):
        
        self.done = False
        self.observation = np.zeros([9, 1])
    
    def step(self, action):
        #1. Update the environment state based on the action chosen
        


        #2. Calculate the "reward" for the new state
        
        #3. Store the new "observation" for the state
        
        #4. Check if the episode is over and store as "done"
        
        
        return self.observation, reward, self.done

    def reset(self):
        
        self.done = False
        self.observation = np.zeros([9, 1])
        
        return self.observation
    
     def render(self):
        
        def letter(number):
        
            if number == 0:
                mark = " "
            elif number < 0:
                    mark = "O"
            elif number > 0:
                    mark = "X"
                
            return mark