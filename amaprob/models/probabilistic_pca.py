import numpy as np
from scipy.stats import multivariate_normal

class PPCA:
    def __init__(self, X: np.ndarray, latent_dim: int) -> None:
        self.N, self.D = X.shape
        self.L = latent_dim 
        
        self.X = X

        self.mu = X.mean(axis=0)
        self.sigma = 1.0
        self.W = np.random.rand(self.D, self.L)  # (D, L)
        
        self.E_z = None
        self.E_zzT = None

    
    def __step_e(self) -> None:
        M = self.W.T@self.W + self.sigma*np.identity(self.L)
        M_inv = np.linalg.inv(M)
        
        self.E_z = (self.X - self.mu) @ (M_inv@self.W.T).T

        self.E_zzT = np.array([
            self.sigma * M_inv + self.E_z[i].reshape(-1, 1) * self.E_z[i].reshape(-1, 1).T
        for i in range(self.N)])
    

    def __step_m(self) -> None:
        left_w = np.array([(self.X[i] - self.mu).reshape(-1, 1) * self.E_z[i].reshape(-1, 1).T for i in range(self.N)]).sum(axis=0)
        right_w = np.linalg.inv(self.E_zzT.sum(axis=0))
        self.W = left_w @ right_w

        aux_sigma = np.array([
            np.linalg.norm(self.X[i] - self.mu) ** 2 - 2*self.E_z[i].reshape(-1, 1).T @ self.W.T @ (self.X[i] - self.mu) + np.trace(self.E_zzT[i]@self.W.T@self.W)
        for i in range(self.N)])
        self.sigma = aux_sigma.sum()/(self.N * self.D)

    
    def fit(self, num_epochs: int = 100) -> None:
        self.__step_e()
        self.__step_m()


    def sample(self) -> np.ndarray:
        random_latent_sample = np.random.rand(self.L, 1)
        mean = (self.W@random_latent_sample + self.mu.reshape(-1, 1)).reshape(1, -1)[0]
        cov = 0.0 * np.identity(self.D)

        return multivariate_normal.rvs(mean, cov)
