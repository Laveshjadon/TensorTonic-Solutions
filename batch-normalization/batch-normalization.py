import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).
    
    """
    x = np.asarray(x)
    gamma = np.asarray(gamma)
    beta = np.asarray(beta)
    if x.ndim == 2:
        mu  = x.mean(axis = 0)
        var = x.var(axis = 0)
        x_hat = (x-mu)/ np.sqrt(var + eps)
        out = gamma * x_hat + beta
    elif x.ndim == 4:
        mu = x.mean(axis =(0,2,3), keepdims = True)
        var = x.var(axis=(0,2,3), keepdims = True)
        x_hat = (x - mu) / np.sqrt(var + eps)
        out = gamma.reshape(1,-1,1,1) * x_hat + beta.reshape(1,-1,1,1)
    else:
        raise ValueError
    return out 
        
    