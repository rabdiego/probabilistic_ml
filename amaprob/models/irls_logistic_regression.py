import numpy as np
from scipy.special import expit

class IRLSLogisticRegression:
    def __init__(self, means: float, covariance: float) -> None:
        self.means = means
        self.covariance = covariance
        self.w = means


    def fit(self, X: np.ndarray, y: np.ndarray, thr: float = 1e-3) -> None:
        err = 1.0
        n, d = X.shape
        while err > thr:
            w_p = self.w
            R = np.diag([float((expit(self.w.T @ X[i])*(1 - expit(self.w.T @ X[i])))[0]) for i in range(n)])
            A = X.T@R@X + self.covariance

            self.w = w_p + np.linalg.inv(A) @ (X.T @ (y - expit(X@w_p)) - np.linalg.inv(self.covariance)@(w_p - self.means))

            err = np.linalg.norm(self.w - w_p)


    def predict(self, X: np.ndarray) -> np.ndarray:
        return expit(X@self.w)

