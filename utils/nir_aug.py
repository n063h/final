import numpy as np
from scipy.interpolate import CubicSpline

class BaseTransform:
    def __init__(self, fn,alpha,beta):
        self.fn=fn
        self.alpha = alpha
        self.beta = beta
        
    def __call__(self, x):
        return self.fn(x, self.alpha, self.beta)

def DA_Jitter(X, sigma=0.05,loc=0):
    myNoise = np.random.normal(loc=loc, scale=sigma, size=X.shape)
    return X+myNoise

def DA_Scaling(X, sigma=0.1,loc=1):
    scalingFactor = np.random.normal(loc=loc, scale=sigma, size=(1))
    myNoise = np.matmul(np.ones((X.shape[0],1)), scalingFactor)
    return X*myNoise

def GenerateRandomCurves(X, sigma=0.1, knot=4):
    xx = (np.ones((X.shape[1],1))*(np.arange(0,X.shape[0], (X.shape[0]-1)/(knot+1)))).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot+2, X.shape[1]))
    x_range = np.arange(X.shape[0])
    cs_x = CubicSpline(xx[:,0], yy[:,0])
    return np.array([cs_x(x_range)]).transpose()

def DA_MagWarp(X, sigma=0.05,knot=4):
    return (X.reshape((228,1)) * GenerateRandomCurves(X.reshape((228,1)), sigma,knot)).reshape((228))

def do_nothing(x, sigma=0.05,knot=4):
    return x

aug_fn={
    'jitter':DA_Jitter,
    'scaling':DA_Scaling,
    'magwarp':DA_MagWarp,
    'do_nothing':do_nothing
}

def build_augs(augs,_alpha=None,_beta=None):
    res=[]
    
    for aug in (augs or []):
        name,alpha,beta=aug.name,aug.alpha,aug.beta
        if _alpha:
            alpha=_alpha
        if _beta:
            beta=_beta
        res.append(BaseTransform(aug_fn[name],alpha,beta))
    return res