import gym
import keras
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from keras.models import Model
from keras.layers import Input, Dense,Conv2D
from keras.optimizers import Adam

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

# remainings are the original one

class  DQN_agent:

    def __init__(self,env):
        self.env=env
        self.state_size=env.observation_space.shape[0]
        self.action_size=env.env.action_space.n
        self.Memory=deque(maxlen=2000)
        self.gamma=0.95
        self.epsilon=1.0
        self.epsilon_min=0.01
        self.epsilon_decay_r = 0.995
        self.learning_rate = 0.01
        self.model=self._build_model()
    
    def _build_model(self): # value function
        
        '''
        Used for Value Function Approximation.
        The input state for CartPole game is quite simple, 
        thus the model I'm using just have two hidden layers with no 
        convolutional layer. 
        '''

        Inputs= Input(shape=(self.state_size,))
        X= Dense(24,activation='relu')(Inputs)
        X= Dense(24,activation='relu')(X)
        X= Dense(self.action_size)(X)
    
        model = Model(inputs=Inputs, outputs=X)
        model.compile(loss='mse',optimizer=Adam(lr=self.learning_rate))

        return model

    def remember(self ,state, action, reward, next_state, done): # memory
        '''
        Remerber the current environment infomation by adding it into the Memory que.
        '''
        self.Memory.append((state, action, reward, next_state, done))
    
        return

    def replay(self,batch_size): # if len(Memory) > batch_size: # train the model after the Memory is enough
        '''
        Replay means sample data from the Memory and use it to train our model
        after adding labels computed by Bellman equation.
        '''

        minibatch = random.sample(self.Memory, batch_size) # 從memory sample batch_size大小的
        States = np.empty((batch_size, self.state_size)) ## states
        Labels = np.empty((batch_size, self.action_size)) ## labels
        
        i = 0
        
        for state, action, reward, next_state, done in minibatch:
    
            if not done: # When the game is not finished
                # Bellman equation to computer the opimal Q-function
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
                
            else:
                # When game ended the Q-funtion is just the current reward
                target = reward

            # Only introduce loss for the action taken this timestep ???
            label = self.model.predict(state)
            label[0][action] = target
            
            # construct the minibatch
            States[i,:] = state
            Labels[i,:] = label

            i += 1
            
        self.model.fit(States, Labels , epochs=1, verbose=0) # Train the model with each state in the minibatch
        self.epsilon_decay() ###
        return


    def act(self,state): # action
        '''
        Randomly select an action with a probability of epsilon
        or act to maximize the Q-Function value estimated by the model 
        '''
    
        if np.random.rand() <= self.epsilon: # Randomly select an action
            return self.env.action_space.sample()
    
        act_values = self.model.predict(state)
            
        return np.argmax(act_values[0]) # maximize the Q-Function value


    def epsilon_decay(self): # epsilon decay
        '''
        The agent should be more dependent on our model after training for some time.
        We can do this by decreasing the probability to choose random actions

        self.epsilon=1.0
        self.epsilon_min=0.01
        self.epsilon_decay_r = 0.995
        '''

        if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay_r
            
        return


def train(agent, Episodes=500, batch_size = 32):

    done = False
    agent.env._max_episode_steps = 1000
    Score = []
    
    for e in range(Episodes): 
    # e stands for the e-th restarted game

        # reshape the state for the keras model to accept
        state = np.reshape(agent.env.reset(), [1, agent.state_size])
        
        for time in range(20000):
            # agent takes an action in each time untill done

            # if e%50 == 0:
                # agent.env.render() # update the frame for the game window 

            action = agent.act(state) # get the action by calling the funtion above
            
            next_state, reward, done, _ = agent.env.step(action) # return np.array(self.state), reward, done, {}
            
            # reward = reward if not done else 1 # the agent is design to last long enough in the game, so the reward is negative when done  ???
            
            next_state = np.reshape(next_state, [1, agent.state_size])

            agent.remember(state, action, reward, next_state, done) # memory
            
            state = next_state # update
            
            if done: # restart the game if done
                print("episode: {}/{}, score: {}, epsilon: {:.2}".format(e, Episodes, time, agent.epsilon))
                Score.append(time) # define the score as the number of actions taken
                break
                
            if len(agent.Memory) > batch_size: ###???
                # train the model after the Memory is enough ???
                agent.replay(batch_size)
                
    return Score


if __name__ =="__main__":
    # environment
    Env = gym.make('CartPole-v1')
    
    # agent
    agent = DQN_agent(Env)

    # score
    score = train(agent)
    
    plt.plot(range(500), score)
    plt.ylabel('score')
    plt.xlabel('episodes')
    plt.show()