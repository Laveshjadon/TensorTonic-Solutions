import numpy as np

def relu(x):
    x = np.array(x)
    result = np.maximum(0,x)
    return result