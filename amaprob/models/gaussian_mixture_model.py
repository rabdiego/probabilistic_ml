import numpy as np

class GaussianMixtureModel:
    def __init__(self, K: int, alpha, mu_o, kapa_o: float, v_o: float, S_o, pi, mu, sigma) -> None:
        self.K = K

        self.v_o = v_o
        self.kapa_o = kapa_o

        # All of the variables below are np.ndarray
        self.alpha = alpha  # (1, K)
        self.mu_o = mu_o  # (1, D)
        self.S_o = S_o
        self.pi = pi  # (1, D)
        self.mu = mu  # (K, D)
        self.sigma = sigma  # (K, D, D)

        # self.r = np.zeros
    

    def __step_e(self, x: np.ndarray) -> None:
        pass


    def fit(self, X: np.ndarray) -> None:
        pass


    def predict(self, X: np.ndarray) -> np.ndarray:
        # should return a column vector of shape (K, 1)
        
        pass