import gym
from gym.envs.registration import register
import readchar
import msvcrt
import numpy as np
import random as pr
import matplotlib.pyplot as plt

class _GetchWindows:
    def __init__(self):
        import msvcrt

    def __call__(self):
        import msvcrt
        return msvcrt.getch()


getch = _GetchWindows()

#Macro
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

# key mapping
arrow_keys ={
    b'w' : UP,
    b's' : DOWN,
    b'd' : RIGHT,
    b'a' : LEFT}


def rargmax(vector):
    """Argmax that chooses randomly among eligible maxium indices"""
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return pr.choice(indices)

#register frozenlake with is_slippery false
register(
    id = 'FrozenLake-v3',
    entry_point = 'gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False}
)

env = gym.make('FrozenLake-v3')

Q = np.zeros([env.observation_space.n, env.action_space.n])
dis = .99
num_episodes = 2000

rList = []
for i in range(num_episodes):
    state = env.reset()
    rAll = 0
    done = False

    #The Q-Table learning algorithm
    while not done:

        action = np.argmax(Q[state, :] + np.random.randn(1,env.action_space.n) / (i + 1))
        #Get new state and reward from environment
        new_state, reward, done, _  = env.step(action)

        #Update Q-Table with new knowledge using learning rate
        Q[state, action] = reward + dis * np.max(Q[new_state,:])

        rAll += reward
        state = new_state

    rList.append(rAll)

print("Success rate: " + str(sum(rList) / num_episodes))
print("Final Q-Table Values")
print(Q)
plt.bar(range(len(rList)), rList, color="blue")
plt.show()
