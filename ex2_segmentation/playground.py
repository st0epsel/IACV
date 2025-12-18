import numpy as np

x = np.arange(25)
x2 = np.reshape(x, (5, 5))

print(np.roll(x2, 1))

print(np.roll(x2, -1))

print(np.roll(x2, 1, axis=0))

print(np.roll(x2, -1, axis=0))

print(np.roll(x2, 1, axis=1))

print(np.roll(x2, -1, axis=1))

print(np.roll(x2, (1, 1), axis=(1, 0)))

print(np.roll(x2, (2, 1), axis=(1, 0)))

print(np.roll(x2, (1, 2), axis=(0, 1)))

