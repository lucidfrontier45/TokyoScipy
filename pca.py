# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 19:08:18 2011

@author: du
"""

import numpy as np
from scipy.linalg import eigh, svd
    
class PCA():
    def __init__(self,whiten=False):
        self._whiten = False
        
    def fit(self,X,method="evd"):
        
        # preprocessing data
        self.m = np.mean(X,0)
        X = X - self.m

        # scale coefficient so that 
        if self._whiten:
            v = np.std(X,0,ddof=0)
            X = X / v
            
        # performe matrix decomposition        
        if method.lower() == "evd":
            
            # use eigen value decomposition
            cv = np.cov(X.T,ddof=0)
            eig_val, eig_vec = eigh(cv)

            # sort in decreasing order
            eig_order = np.argsort(eig_val)[::-1]
            self.var = eig_val[eig_order]
            self.coeff = eig_vec[:,eig_order]

        elif method.lower() == "svd":                

            # use singular value decomposition
            u, s, w = svd(X,full_matrices=False)
            self.var = s**2 / len(X)
            self.coeff = w.T

        else:
            raise  ValueError, "unknown PCA method"
        
    def transform(self,X):
        return np.dot(X-self.m, self.coeff)
        
    def var_ratio(self):
        return  self.var / self.var.sum()
            
            
if __name__ == "__main__":
    from scikits.learn.datasets import load_iris
    import pylab
   
    iris = load_iris()  
    model = PCA()
    model.fit(iris.data)
    pc = model.transform(iris.data)
    for t in range(3):
        mask = (iris.target == t)
        pylab.plot(pc[mask,0],pc[mask,1],"o",label=iris.target_names[t])
    pylab.legend()
    pylab.show()