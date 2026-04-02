import numpy as np

def linear_regression_closed_form(X, y):
   
    w = np.linalg.pinv(X)@y
    return w