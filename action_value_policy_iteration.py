import matplotlib.text
import numpy as np
import matplotlib.pyplot as plt
from collections import *

grid_world = np.arange(16).reshape(4, 4)
values = np.array(grid_world, dtype=str)
text_array = [0] * 16
state_action_state = namedtuple("sas", ["state", "action_next_state"])
action_space = np.arange(4)
states = []
for (i, j), v in np.ndenumerate(grid_world):
    if i - 1 >= 0:
        up = [0, v - 4]
    else:
        up = [0, v]
    if i + 1 <= 3:
        down = [1, v + 4]
    else:
        down = [1, v]
    if j - 1 >= 0:
        left = [2, v - 1]
    else:
        left = [2, v]
    if j + 1 <= 3:
        right = [3, v + 1]
    else:
        right = [3, v]
    action_next_states = [up, down, left, right]
    states.append(state_action_state(v, action_next_states))

for s in states:
    print(s.state, s.action_next_state)

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
