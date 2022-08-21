"""
Action value policy iteration on 4x4 grid world
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import *
from itertools import *

grid_world = np.arange(16).reshape(4, 4)
values = np.array(grid_world, dtype=str)
text_array = [0] * 16
state_action_next_state = namedtuple("sas", ["state_action", "next_state"])
action_space = np.arange(4)
state_action = product(grid_world.reshape(-1), action_space)
states = []

'''Generate state-action,next-state named tuple'''
for sa in state_action:
    row = sa[0] // 4
    col = sa[0] % 4
    next_state = sa[0]
    if sa[1] == 0:
        if row - 1 >= 0:
            next_state = sa[0] - 4
    if sa[1] == 1:
        if row + 1 <= 3:
            next_state = sa[0] + 4
    if sa[1] == 2:
        if col - 1 >= 0:
            next_state = sa[0] - 1
    if sa[1] == 3:
        if col + 1 <= 3:
            next_state = sa[0] + 1
    states.append(state_action_next_state(sa, next_state))

'''Initialization of q-values and policy'''
qvalues = np.zeros(len(states))
policy = np.ones((16, 4)) * 0.25


def evaluate(states=states, qvalues=qvalues, policy=policy):
    for i, (state_action, next_state) in enumerate(states):
        if 3 < i < 60:
            indices = [states.index(sans) for sans in states if next_state == sans.state_action[0]]
            qvalues[i] = policy[next_state] @ (-1 + qvalues[indices])
    return qvalues


def update_policy(qvalues):
    shape = (16, 4)
    max_index = np.argmax(qvalues.reshape(shape), axis=1)
    new_policy = np.zeros(shape)
    new_policy[np.arange(shape[0]), max_index] = 1
    return new_policy


'''Showing state value changes graphcially'''


def canvas(states, qvalues, policy, state_values, episodes=10):
    grid_world_shape = (4, 4)
    with plt.ion():
        plt.matshow(grid_world)
        plt.axis("off")
        plt.show()
        for (i, j), v in np.ndenumerate(state_values):
            text_array[i * 4 + j] = plt.text(i, j, str(v))
        for e in range(episodes):
            qvalues = evaluate(states, qvalues, policy)
            policy = update_policy(qvalues)
            state_values = np.sum(qvalues.reshape(policy.shape) * policy, axis=1).reshape(grid_world_shape)
            for (i, j), v in np.ndenumerate(state_values):
                text_array[i * 4 + j].set(text=str(np.round(v, 2)))
            plt.pause(1)


state_values = np.zeros(grid_world.shape)
canvas(states, qvalues, policy, state_values, 100)
