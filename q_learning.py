import gym
import random
import numpy
import sys

env = gym.make('CartPole-v0')


resolution = 8
bound = 2.5
epsilon = 1
epsilon_decrease_factor = 0.997
learning_speed = 1
learning_speed_decrease_factor = 0.999


policy = None

def sample(value, low, high):
    low = max(low,-1*bound)
    high = min(high,bound)
    if value < -1*bound :
        value = -1*bound
    if value > bound:
        value = bound
    return min((resolution *(value - low) / (high - low)), resolution-1)

def get_state(observation):
        result = []
        for i in range(env.observation_space.shape[0]):
            value = observation[i]
            low = env.observation_space.low[i]
            high = env.observation_space.high[i]
            result.append(int(sample(value, low, high)))
        return(result)
        
def get_state_value(state):
        result = policy[:]
        for i in state:
            result = result[i]
        return(result)
        
def set_state_value(state, value):
        global policy
        policy[tuple(state)] = value
        return
        
def init_policy(resolution):
    dimensions = ()
    for i in range(env.observation_space.shape[0]):
        dimensions = dimensions + (resolution,)
    dimensions = dimensions + (env.action_space.n,)
    #policy = numpy.random.normal(0,1,dimensions)
    policy = numpy.zeros(dimensions)
    print(dimensions)
    return(policy)
        

def learn(render_rate):
    global policy
    global epsilon
    global epsilon_decrease_factor
    global learning_speed
    global learning_speed_decrease_factor
    
    policy = init_policy(resolution)
    
    for i_episode in range(200000):
        
        observation = env.reset()
    
        epsilon = epsilon_decrease_factor * epsilon
        #epsilon=max(epsilon,0.05)
        
        learning_speed = 0.999 * learning_speed
        #learning_speed = max(0.005,learning_speed)
        
        #print(epsilon)
            
        for t in range(500):
            
            if (i_episode%render_rate==0):
                env.render()
            
            state = get_state(observation)
            state_values = get_state_value(state)
        
            value = max(state_values)


            if numpy.random.random() > epsilon:
                action = numpy.argmax(state_values,axis=0)
            else:
                action = env.action_space.sample()
    
    
    
            observation, reward, done, info = env.step(action)
    
            new_state = get_state(observation)
            new_state_values = get_state_value(new_state)
                    

            learnt_value = learning_speed * (reward + 0.9*(max(new_state_values) - state_values[action]))
            

            
            state.append(action)
            set_state_value(state,state_values[action] + learnt_value)
            
            
            
            
            if done:
                print("Episode {} finished after {} timesteps".format(i_episode,t+1))
                break
    return
            
learn(render_rate = 500)

