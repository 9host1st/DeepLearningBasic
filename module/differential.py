import numpy as np

class Differential:
    def numericalDifferential(f, x):
        h = 1e-7
        return (f(x + h) - f(x - h)) / 2*h
    
    def numericalGradient(f, x):
        h = 1e-7
        gard = np.zeros_like(x)

        for idx in range(x.size):
            tmpVal = x[idx]
            x[idx] = tmpVal + h
            fxh1 = f(x)
            x[idx] = tmpVal - h
            fxh2 = f(x)

            gard[idx] = (fxh1 - fxh2) / (2*h)
            x[idx] = tmpVal
        return grad

    def gradientDescent(f, initX, lr, step):
        x = initX

        for i in range(step):
            gard = numericalGradient(f, x)
            x -= lr * grad
        return x
