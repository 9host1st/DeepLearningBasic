import numpy as np

class Output:
    def identityFunction(x):
        return x

    def softmax(x):
        c = np.max(x)
        expA = np.exp(x - c)
        sumExpA = np.sum(expA)
        return expA / sumExpA
