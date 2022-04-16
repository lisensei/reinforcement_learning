import numpy as np
from collections import *

num_grid = 16
num_actions = 4
policy = np.zeros((num_grid, num_actions))
grid_world = np.arange(num_grid).reshape(-1, int(num_grid ** 0.5))
state = namedtuple("state", ["state", "next_states", "prob"])


def generate_states():
    states = []
    for (row, col), value in np.ndenumerate(grid_world):
        next_states = np.ones(4) * value
        prob = np.ones(4) * 0.25
        if row - 1 >= 0:
            next_states[0] = value - 4
        if row + 1 <= 3:
            next_states[1] = value + 4
        if col - 1 >= 0:
            next_states[2] = value - 1
        if col + 1 <= 3:
            next_states[3] = value + 1
        states.append(state(value, next_states, prob))
    return states


def evaluate_policy(policy=policy):
    state_values = np.zeros(num_grid)
    return state_values


def update_policy(policy):
    return policy


dic = {0: "up", 1: "down", 2: "left", 3: "right"}
states_array = generate_states()

for _, s in enumerate(states_array):
    print(s.state, s.next_states, s.prob)
