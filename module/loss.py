import numpy as np

class Loss:
    def meanSquareError(y, t):
        return 0.5 * np.sum((y - t) ** 2)

    def crossEntropyError(y, t):
        delta = 1e-10
        return -np.sum(t*np.log(y + delta))

