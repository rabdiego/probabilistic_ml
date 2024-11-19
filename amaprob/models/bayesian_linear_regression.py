import numpy as np

class BayesianLinearRegression:
    def __init__(self, mean_prior: np.ndarray, covariance_prior: np.ndarray, noise: float = 1.0) -> None:
        self.mean = mean_prior
        self.covariance = covariance_prior
        self.noise = noise
        self.d = mean_prior.shape[0]


    def update(self, X: np.ndarray, y: np.ndarray) -> None:
        mean_prior = self.mean
        covariance_prior = self.covariance
        
        aux = np.linalg.inv(covariance_prior@X.T@X + self.noise**2 * np.identity(self.d))
        self.mean = mean_prior + aux @ covariance_prior@X.T@(y - X@mean_prior)
        self.covariance = covariance_prior - aux @ covariance_prior@X.T@X@covariance_prior
            

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X@self.mean


    def get_std(self, X: np.ndarray, split: bool = False) -> np.ndarray:
        n = X.shape[0]
        if not split:
            return np.array([(X @ self.covariance @ X.T + self.noise**2 * np.identity(n)).diagonal()]).T
        return np.array([(X @ self.covariance @ X.T).diagonal()]).T, np.array([(self.noise**2 * np.identity(n).diagonal())]).T

