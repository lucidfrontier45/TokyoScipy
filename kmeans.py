# -*- coding: utf-8 -*-
"""
Created on Wed Aug 03 12:44:03 2011

@author: Du
"""

import numpy as np
from numpy.random import randint
from scipy.spatial.distance import cdist

class KMeans():
    def __init__(self, nclust=5):
        self.nclust = nclust
        
    def _initialize(self, X):
        nclust = self.nclust
        nobs, ndim = X.shape
        
        self.mu = np.empty((nclust,ndim))
        codes = randint(0,nclust,nobs)
        
        for c in xrange(nclust):
            mask = (codes == c)
            self.mu[c] = np.mean(X[mask],0)
        
    def _E_step(self, X, dist_mat=None):
        nclust = self.nclust
        nobs, ndim = X.shape

        if dist_mat == None:
            dist_mat = np.empty((nobs,nclust))
         
        dist_mat = cdist(X,self.mu)

        score = (dist_mat.min(1)**2).sum()
        codes = dist_mat.argmin(1)
        
        return codes, score
        
    def _M_step(self, X, codes):
        nclust = self.nclust
        for c in xrange(nclust):
            mask = (codes == c)
            if mask.sum() == 0:
                self.mu[c] = 1.0e50
            else:
                self.mu[c] = np.mean(X[mask],0)
        
    def fit(self, X, maxiter=100, thrs=1.0e-4, init=True):
        nclust = self.nclust
        nobs, ndim = X.shape

        if init:
            self._initialize(X)
            
        dist_mat = np.empty((nobs,nclust))
        score = 1.0e50
        for i in xrange(maxiter):
            # performe E step            
            codes, new_score = self._E_step(X,dist_mat)
            
            # checking convergence            
            dscore = new_score - score
            if dscore > -thrs:
                print "%5dth iter, score = %8.3e, dscore = %8.3e > %8.3e converged" \
                    % (i, new_score, dscore, -thrs)
                break
            print "%5dth iter, score = %8.3e, dscore = %8.3e" \
                % (i, new_score, dscore)
            score = new_score

            # performe M step
            self._M_step(X,codes)    
    
    def decode(self,X):
        codes, score = self._E_step(X)
        return codes
    