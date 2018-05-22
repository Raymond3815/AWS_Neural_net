import numpy as np


t = np.concatenate((
    [(1, 0) for _ in range(5)], [(0, 1) for _ in range(7)]
))
print(t)
