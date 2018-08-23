import numpy as np

class MatrixFactorization(object):
    def __init__(self, data, k):
        '''
        Arguments:
        - data    : 2 dimensional rating matrix
        - k       : number of latent dimensions
        '''
        
        self.R = np.matrix(data)
        self.D = np.zeros( self.R.shape )
        self.k = k
        
        self.U = 2*(np.random.uniform( size=(self.R.shape[0], k) )-.5)
        self.P = 2*(np.random.uniform( size=(k, self.R.shape[1]) )-.5)
    
    def _compure_error(self):
        self.D = (self.R - self.estimate_all())
        
        return self.D
    
    def train(self, alpha=0.1, beta=0.01, iterations=1000):
        '''
        Arguments:
        - alpha   : learning-rate 
        - beta    : regularization-rate
        '''
        
        for _ in range(iterations):
            self._compure_error()
            
            U = self.U
            P = self.P
            
            for i in range(self.R.shape[0]):      
                for j in range(self.R.shape[1]):
                    for k in range(self.k):
                        ik = (alpha/self.k) * P[k, j] * self.D[i, j] 
                        kj = (alpha/self.k) * U[i, k] * self.D[i, j]
                        if not np.isnan(ik):
                            self.U[i, k] += ik
                        if not np.isnan(kj):
                            self.P[k, j] += kj
    
    def estimate_all(self):
        return self.U.dot(self.P)
    
    def estimate(self, x, y):
        return self.U[x, :].dot(self.P[:, y])
