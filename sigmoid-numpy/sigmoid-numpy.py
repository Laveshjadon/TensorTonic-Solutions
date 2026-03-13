import numpy as np

def sigmoid(x):
    x = np.array(x)
    result = 1 / (1 + np.exp(-x))
    return result.tolist()