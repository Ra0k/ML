import numpy as np

class MatrixFactorization(object):
    def __init__(self, data, K):
        '''
        Arguments:
        - data    : 2 dimensional rating matrix
        - K       : number of latent dimensions
        '''
        
        self.R = np.matrix(data)
        self.D = np.zeros( self.R.shape )
        self.K = K
        
        self.U = np.random.uniform( size=(self.R.shape[0], K) )
        self.P = np.random.uniform( size=(K, self.R.shape[1]) )
    
    def _compure_error(self):
        self.D = (self.R - self.estimate_all())
        
        return self.D
    
    def train(self, alpha=0.1, beta=0.02, iterations=1000):
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
                    for k in range(self.K):
                        ik = alpha * ( P[k, j] * self.D[i, j] + beta * U[i, k])
                        kj = alpha * ( U[i, k] * self.D[i, j] + beta * P[k, j])
                        if np.isfinite(ik):
                            self.U[i, k] += ik
                        if np.isfinite(kj):
                            self.P[k, j] += kj
            
            #non-negativity
            self.U = self.U.clip(min=0)
            self.P = self.P.clip(min=0)
            
        return np.nansum(np.nansum(abs(self._compure_error())))
    
    def estimate_all(self):
        return self.U.dot(self.P)
    
    def estimate(self, x, y):
        return self.U[x, :].dot(self.P[:, y])
