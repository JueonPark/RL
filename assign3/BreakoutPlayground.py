import gym
import numpy as np
from matplotlib import pyplot as plt

env = gym.envs.make("Breakout-v0")

print("Action space size: {}".format(env.action_space.n))
print(env.get_action_meanings()) # env.unwrapped.get_action_meanings() for gym 0.8.0 or later

observation = env.reset()
print("Observation space shape: {}".format(observation.shape))

plt.figure()
plt.imshow(env.render(mode='rgb_array'))
plt.show()

[env.step(1) for x in range(2)]
plt.figure()
plt.imshow(env.render(mode='rgb_array'))
plt.show()

# Check out what a cropped image looks like
plt.imshow(observation[34:-16,:,:])
plt.show()

env.reset()
for _ in range(10000):
    env.render()
    env.step(env.action_space.sample()) # task a random action
env.close()
