import gym

env = gym.make('CartPole-v0')
for i_episode in range(500):
    observation = env.reset()
    for t in range(200):
        env.render() 
        if (observation[1] + observation[2] + observation[3] < 0) :
            action = 0
        else:
            action = 1
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break