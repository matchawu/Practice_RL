import gym
import keras
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from keras.models import Model
from keras.layers import Input, Dense,Conv2D
from keras.optimizers import Adam


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
    
    def _build_model(self):

        Inputs= Input(shape=(self.state_size,))
        X= Dense(24,activation='relu')(Inputs)
        X= Dense(24,activation='relu')(X)
        X= Dense(self.action_size)(X)
    
        model = Model(inputs=Inputs, outputs=X)
        model.compile(loss='mse',optimizer=Adam(lr=self.learning_rate))

        return model

    def remember(self ,state, action, reward, next_state, done):
        self.Memory.append((state, action, reward, next_state, done))
    
        return

    def replay(self,batch_size):
    
        minibatch = random.sample(self.Memory, batch_size)
        States=np.empty((batch_size,self.state_size))
        Labels=np.empty((batch_size,self.action_size))
        i=0
        for state,action,reward,next_state,done in minibatch:
    
            if not done:
                target=reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            else:
                target=reward
        
            label=self.model.predict(state)
            label[0][action]= target
            States[i,:]=state
            Labels[i,:]=label
            i+=1
        self.model.fit(States, Labels , epochs=1, verbose=0)
        self.epsilon_decay()
        return


    def act(self,state):
    
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()
    
        act_values = self.model.predict(state)
            
        return np.argmax(act_values[0]) 


    def epsilon_decay(self):
        if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay_r
            
        return


def train(agent, Episodes=500, batch_size = 32):

    done=False
    agent.env._max_episode_steps = 1000
    Score=[]
    
    for e in range(Episodes):
        state = np.reshape(agent.env.reset(), [1, agent.state_size])
        
        for time in range(20000):
            if e%50 == 0:
                agent.env.render()
            action = agent.act(state)
            next_state, reward, done, _ = agent.env.step(action)
            # reward = reward if not done else 1
            next_state = np.reshape(next_state, [1, agent.state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, epsilon: {:.2}".format(e, Episodes, time, agent.epsilon))
                Score.append(time)
                break
                
            if len(agent.Memory) > batch_size:
                agent.replay(batch_size)
                
    return Score


if __name__ =="__main__":
    Env=gym.make('CartPole-v1')
    agent=DQN_agent(Env)

    score=train(agent)
    
    plt.plot(range(500), score)
    plt.ylabel('score')
    plt.xlabel('episodes')
    plt.show()