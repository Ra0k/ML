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
           
        # User and Product biases
        self.b   = np.nanmean(self.R)
        self.u_b = np.zeros( self.R.shape[0] )
        self.p_b = np.zeros( self.R.shape[1] )
        
        # User and Product matrix
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
            
            for i in range(self.R.shape[0]):      
                for j in range(self.R.shape[1]):
                    for k in range(self.K):
                        #update User and Product matrix
                        U_ik = alpha * ( self.P[k, j] * self.D[i, j] - beta * self.U[i, k])
                        P_kj = alpha * ( self.U[i, k] * self.D[i, j] - beta * self.P[k, j])
                        if np.isfinite(U_ik):
                            self.U[i, k] += U_ik
                        if np.isfinite(P_kj):
                            self.P[k, j] += P_kj
                            
                        #update User and Product biases
                        Ub_i = alpha * ( self.D[i, j] - beta * self.u_b[i] )
                        Pb_j = alpha * ( self.D[i, j] - beta * self.p_b[j] )
                        if np.isfinite(Ub_i):
                            self.u_b[i] += Ub_i
                        if np.isfinite(Pb_j):
                            self.p_b[j] += Pb_j      
            
            
        return np.nansum(np.nansum(abs(self._compure_error())))
    
    def estimate_all(self):
        return self.U.dot(self.P) + self.b + self.u_b[:, np.newaxis] + self.p_b[np.newaxis, :]
    
    def estimate(self, x, y):
        return self.U[x, :].dot(self.P[:, y]) + self.b + self.u_b[x] + self.p_b[y]
