import numpy as np

class Activation:
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def relu(x):
        return np.maximum(0, x)
   
    def stepFunction(x):
        return (x > 0).astype(np.int)
        
