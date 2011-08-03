# -*- coding: utf-8 -*-
"""
Created on Wed Aug 03 12:44:03 2011

@author: Du
"""

import numpy as np
from numpy.random import randint
from scipy.linalg import det,inv
from scipy.spatial.distance import cdist

class AnisotropicKMeans():
    """
    Anisotropic K-Means Class.
    This Class does K-Means clustering with non-unit covariance matrix
    """
    def __init__(self, nclust=5):
        # set maximum number of cluster
        self.nclust = nclust
        
    def _initialize(self, X):
        nclust = self.nclust
        nobs, ndim = X.shape
        
        # allocate mean vectors and covariance matrices
        self.mu = np.empty((nclust,ndim))
        self.cv = np.empty((nclust,ndim,ndim))
        
        # initialize codes with random number
        codes = randint(0,nclust,nobs)
        
        # initialize means and covariances by codes
        for c in xrange(nclust):
            mask = (codes == c)
            self.mu[c] = np.mean(X[mask],0)
            self.cv[c] = np.cov(X[mask].T,ddof=0)            

    def _E_step(self, X, lnP_mat=None):
        """
        In E step, the most probable cluster indexes (codes) are obtained.
        
        Input:
            - X [ndarray, shape(nobs, ndim)] : input data matrix
            - lnP_mat [ndarray, shape(nobs, nclust), optional]
                tempolary space for calculating score
        Output:
            - codes [ndarray, shape(nobs)] : most probable cluste indexes
            - score [float] : score of clustering
        """
        nclust = self.nclust
        nobs, ndim = X.shape
        
        # allocate dist_mat if needed
        if lnP_mat == None:
            lnP_mat = np.empty((nobs,nclust))
         
        # calculate dist_mat
        for c in xrange(nclust):
            lnP_mat[:,c] = - np.log(det(self.cv[c])) \
                - cdist(X,self.mu[c][np.newaxis],\
                    "mahalanobis",VI=inv(self.cv[c])).reshape(-1)**2 
        
        # cumulate minimal distances 
        score = lnP_mat.max(1).sum()
        
        # obtain codes that minimize distance
        codes = lnP_mat.argmax(1)
        
        return codes, score
        
    def _M_step(self, X, codes):
        """
        In M step, parameters are infered so as to maximize the score
        Input:
            - X [ndarray, shape(nobs, ndim)] : input data matrix
            - codes [ndarray, shape(nobs)] : most probable cluste indexes
        """
        nclust = self.nclust
        nobs, ndim = X.shape
        
        #main loop
        for c in xrange(nclust):
            mask = (codes == c)
            # ignore empty cluster 
            if mask.sum() == 0:
                self.mu[c] = 1.0e50
                self.cv[c] = np.identity(ndim)
            # update parameters by data in the clusters
            else:
                self.mu[c] = np.mean(X[mask],0)
                self.cv[c] = np.cov(X[mask].T,ddof=0)
        
    def fit(self, X, maxiter=100, thrs=1.0e-4, init=True,sort=True,plot=False):
        """
        Use iterative EM algorithm to learn parameters
        Input:
            - X [ndarray, shape(nobs, ndim)] : input data matrix
            - maxiter [int > 0, optional] : maximum number of iteration
            - thrs [float > 0, optional] : threshold of convergence
            - init [bool, optional] : if initialize parameter
            - sort [bool, optional] : if sort parameters according to cluster size
            - plot [bool, optional] : if plot the result
        """
        nclust = self.nclust
        nobs, ndim = X.shape

        # perform initialization
        if init:
            self._initialize(X)
        
        # allocate tempolary space for calculation of score
        lnP_mat = np.empty((nobs,nclust))
        
        # start EM algorithm
        score = -1.0e50
        for i in xrange(maxiter):
            
            # performe E step            
            codes, new_score = self._E_step(X,lnP_mat)
            
            # checking convergence            
            dscore = new_score - score
            if dscore < thrs:
                print "%5dth iter, score = %8.3e, dscore = %8.3e > %8.3e converged" \
                    % (i, new_score, dscore, thrs)
                break
            print "%5dth iter, score = %8.3e, dscore = %8.3e" \
                % (i, new_score, dscore)
            score = new_score

            # performe M step
            self._M_step(X,codes)
            
        self.pi = np.bincount(codes) / float(nobs)
        
        # sort parameters
        if sort:
            self.sort()
        
        # plot result
        if plot:
            self.plot2d(X)
        
    def sort(self):
        """
        sort in decreasing order
        """
        sort_order = np.argsort(self.pi)[::-1]
        self.pi = self.pi[sort_order]        
        self.mu = self.mu[sort_order]
        self.cv = self.cv[sort_order]
    
    def showParams(self,show=True):
        """
        show parameters
        Input:
            - show [bool, optional] : if print params in stdout
        """
        if show:
            for c in xrange(self.nclust):
                print "#############################################\n"
                print "%3dth cluster, pi = %8.3e" % (c, self.pi[c])
                print "\ncenter is \n", self.mu[c]
                print "\ncovariance is \n", self.cv[c]
                print "\n---------------------------------------------\n"
            
        return self.pi, self.mu, self.cv
    
    def decode(self,X):
        """
        Decode data into cluster indexes
        """
        codes, score = self._E_step(X)
        return codes
        
    def plot2d(self,X,ax1=0,ax2=1):
        """
        2D scatter plot of the clustering result
        """
        import pylab
        
        # plotting symbols
        symbs = "o.hdx+"

        # get codes
        codes = self.decode(X)

        # plot        
        for c in xrange(self.nclust):
            mask = (codes == c)
            symb = symbs[c/6]
            pylab.plot(X[mask,ax1],X[mask,ax2],symb,label="%3dth cluster"%c)
        pylab.legend()
        pylab.show()
               
        
if __name__ == "__main__":
    m = np.array([5,-5.])
    cv = np.array([[4.,1],[1,2]])
    X = np.r_[np.random.randn(1000,2),np.random.multivariate_normal(m,cv,2000)]
    model = AnisotropicKMeans(2)
    model.fit(X,plot=True)
    model.showParams()