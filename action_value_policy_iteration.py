import matplotlib.text
import numpy as np
import matplotlib.pyplot as plt

grid_world = np.arange(16).reshape(4, 4)
values = np.array(grid_world, dtype=str)
text_array = [0] * 16
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
