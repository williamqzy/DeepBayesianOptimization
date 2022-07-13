import numpy as np
from scipy.stats import norm
import warnings

def ei(x,f,y_best,obj_fun = lambda x:x[:,0,:]):
    y_pred = obj_fun(f(x))
    y_mean = np.mean(y_pred,axis=1)
    y_std = np.std(y_pred,axis=1)
    Z = (y_best - y_mean)/y_std
    prob = (y_mean - y.best) / y_std
    i_max = [np.argmax(prob)]
    return x[i_max,:]

def ei_direct(x,f_surrogate,y_best):
    y_mean, y_pred = f_surrogate(x)
    y_mean = y_mean[:,0]
    y_std = y_pred[:,0]
    prob = (y_mean - y_best) * norm.cdf(y_mean - y_best)/y_std + \
            y_std * norm.pdf(y_mean - y_best)/y_std
    i_max = np.argmax(prob)
    return i_max, x[i_max,:]

