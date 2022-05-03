import matplotlib.text
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

qvalues = np.zeros(len(states))
p = np.ones(4).reshape(1, -1) * 0.25


def update(states=states, qvalues=qvalues):
    for i, (state_action, next_state) in enumerate(states):
        if 3 < i < 60:
            indices = [states.index(sans) for sans in states if next_state == sans.state_action[0]]
            qvalues[i] = p @ (-1 + qvalues[indices])


for i in range(1000):
    update(states, qvalues)
    state_values = qvalues.reshape(16, -1) @ p.reshape(-1, 1)

print(f"action values:\n{qvalues.reshape(16, -1)}")
print(f"state values:\n{state_values.reshape(4, -1)}")


def canvas(gw=grid_world):
    with plt.ion():
        plt.matshow(grid_world)
        plt.show()
        for (i, j), _ in np.ndenumerate(grid_world):
            text_array[i * 4 + j] = plt.text(i, j, "0")
        for s in range(10):
            plt.axis("off")
            for (i, j), v in np.ndenumerate(grid_world):
                text_array[i * 4 + j].set(text=str(np.random.randint(0, 200)))
            plt.pause(1)
