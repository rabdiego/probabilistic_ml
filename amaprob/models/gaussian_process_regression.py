import numpy as np
from scipy.spatial.distance import cdist

class GaussianProcessRegressor:
    def __init__(self, noise=1e-2) -> None:
        self.theta = None
        self.K = None
        self.noise = noise
        self.X = None
        self.y = None


    def __rbf_kernel(self, X1, X2):
        try:
            n = len(X1)
        except TypeError:
            n = 1

        if n > 1:
            K = np.zeros((n, n))

            for i in range(n):
                for j in range(n):
                    dists = self.theta[1] * (X1[i] - X2[j])**2
                    K[i][j] = self.noise * np.exp(-0.5 * dists)
        else:
            dists = self.theta[1] * (X1 - X2)**2
            K = self.noise * np.exp(-0.5 * dists)

        return K


    def fit(self, X, y) -> None:
        self.X = X
        self.y = y
        
        self.theta = np.zeros(3)
        
        self.theta[0] = np.var(y)
        self.theta[2] = 0.01*self.theta[0]
        self.theta[1] = np.power(np.var(X), -1)
        
        self.theta = self.theta.T

        self.K = self.__rbf_kernel(X, X)


    def __single_predict(self, x) -> np.ndarray:
        kfs = np.array([
            self.__rbf_kernel(x, self.X[i]) for i in range(self.X.shape[0])
        ]).T

        kss = self.__rbf_kernel(kfs, kfs)[0][0]

        mean = (kfs.reshape(1, -1) @ np.linalg.inv(self.K + self.theta[2]*np.identity(self.K.shape[0])) @ self.y)[0]
        var = kss - (kfs.reshape(1, -1) @ np.linalg.inv(self.K + self.theta[2]*np.identity(self.K.shape[0])) @ kfs)[0]
        
        return mean, var

    

    def predict(self, X):
        n = X.shape[0]

        meanvar = np.array([
            self.__single_predict(X[i]) for i in range(n)
        ])

        means = meanvar[:, 0]
        stds = np.power(meanvar[:, 1], 0.5)
        return means, stds
    
    