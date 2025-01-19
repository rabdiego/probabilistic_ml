import numpy as np
from scipy.stats import norm, gamma

class GaussianVariationalInference:
    def __init__(self, m_0: float, k_0: float, a_0: float, b_0: float) -> None:
        self.m_0 = m_0
        self.k_0 = k_0
        self.a_0 = a_0
        self.b_0 = b_0

        self.m_n = None
        self.k_n = 1
        self.a_n = None
        self.b_n = 1

        self.m = None
        self.t = None


    def fit(self, x: np.ndarray, num_epochs: int = 10) -> None:
        self.x = x
        self.n = x.shape[0]
        self.__optimal_values()

        for _ in range(num_epochs):
            self.k_n = (self.k_0 + self.n) * (self.a_n / self.b_n)
            self.b_n = self.b_0 + (self.k_0/2)*(1/self.k_n + (self.m_n - self.m_0)**2) + 0.5*(1/self.k_n + np.power(x - self.m_n, 2)).mean()
        
        self.m = self.m_n
        self.t = self.a_n / self.b_n
    

    def __optimal_values(self) -> None:
        self.m_n = (self.k_0 * self.m_0 + self.x.mean())/(self.k_0 + self.n)
        self.a_n = self.a_0 + (self.n + 1)/2
    
