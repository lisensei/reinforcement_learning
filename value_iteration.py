"""
reinforcement learning value iteration algorithm
on grid world to compute optimal state values.
"""
import numpy as np
from collections import namedtuple


state = namedtuple("state", ["state", "next_states"])
grid_world = np.arange(0, 16).reshape(4, 4)
state_values = np.zeros(16)
states = []

'''compute next states reachable from a given state'''
for (i, j), v in np.ndenumerate(grid_world):
    next_states = [v] * 4
    if i - 1 >= 0:
        next_states[0] = v - 4
    if i + 1 <= 3:
        next_states[1] = v + 4
    if j - 1 >= 0:
        next_states[2] = v - 1
    if j + 1 <= 3:
        next_states[3] = v + 1
    states.append(state(state=v, next_states=next_states))


'''value iteration'''
n = 1000
for i in range(n):
    for j in range(1, 15):
        nsv = np.max(state_values[states[j].next_states] - 1)
        state_values[j] = nsv

print(state_values.reshape(-1, 4))
