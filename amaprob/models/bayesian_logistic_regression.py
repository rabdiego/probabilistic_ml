import numpy as np
from scipy.special import expit
from scipy.stats import norm

class BayesianLogisticRegression:
    def __init__(self, means_prior: float, cov_prior: float) -> None:
        self.means = means_prior
        self.covariance = cov_prior


    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n, d = X.shape

        R = np.diag([float((expit(self.means.T @ X[i])*(1 - expit(self.means.T @ X[i])))[0]) for i in range(n)])
        H = X.T@R@X + np.linalg.inv(self.covariance)
        self.covariance = np.linalg.inv(H)


    def predict(self, X: np.ndarray) -> np.ndarray:
        n, _ = X.shape

        mu_a = X@self.means
        sigma_a = X@self.covariance@X.T
        
        return expit(np.power(np.identity(n) + (np.pi/8)*np.power(sigma_a, 2), 0.5) @ mu_a)

