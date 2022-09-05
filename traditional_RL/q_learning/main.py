from gridworld import *
from agent import *

plt.ion()
env = GridWorld(16, 4)
agent = Agent(env, 16, 4)
agent.q_learning(10000)
