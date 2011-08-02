# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 22:43:02 2011

@author: du
"""

import numpy as np
from scipy.linalg import svd

class _BaseRegression():
    """
    Abstruct Class for Regression
    """
    def makeDataMatrix(self,x):
        """
        override this function when implement your own Regression Class
        """
        pass
    
    def fit(self,x,y,plot=False,variable=1,lw=5):
        """
        Calculate regression coefficients by Singular Value Decomposition 
        """
        # transform observed raw data to data design matrix
        X = self.makeDataMatrix(x)
        # compute coefficients by SVD
        u, s, w = svd(X,full_matrices=False)
        self._coeff = np.dot(w.T*(s**-1)[np.newaxis,:],np.dot(u.T,y))
        # evaluate fitting error        
        yy = np.dot(X,self._coeff)
        residual_variance = np.sum((y-yy)**2)
        self._rmsd_error = np.sqrt(residual_variance / len(y))
        print "fitting rmsd error = ", self._rmsd_error
        print "coefficients are"
        print self._coeff
        # plot fitting result if needed
        if plot:
            try:
                import pylab
            except ImportError:
                print "matplotlib was not found"
                print "plotting is omitted"
                return
            if x.ndim > 1:
                xx = x[:,variable]
            else:
                xx = x
            pylab.plot(xx,y,"o",label="observed")
            pylab.plot(xx,yy,linewidth=lw,label="predicted")
            pylab.legend()
            pylab.show()
        
    def predict(self,x):
        """
        Predict value of y
        """
        X = self.makeDataMatrix(x)
        return np.dot(X,self._coeff)
        
        
class PolynomialRegression(_BaseRegression):
    """
    Polynomial Regression Class
    y = a_0 + a_1*x + a_2*x^2 + ...
    """
    def __init__(self,degree=2):
        self.degree = degree
        
    def makeDataMatrix(self,x):
        X = np.empty((len(x),self.degree))
        for d in xrange(self.degree):
            X[:,d] = x**d
        return X
        
class SineRegression(_BaseRegression):
    def __init__(self,freq):
        self.freq = freq
      
    def makeDataMatrix(self,x):
        X = np.empty((len(x),len(self.freq)+1))
        X[:,0] = 1.0
        for d,f in enumerate(self.freq):
            X[:,d+1] = np.sin(x*f)
        return X
    
        
        