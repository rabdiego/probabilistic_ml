"""
Professor, escrevo este comentário com um sentimento profundo de vergonha no meu coração,
pois no momento em que a escrevo, também deixo por completamente incompleto a lista 3.
Por burrice de minha parte, deixei pra fazê-la na última semana possível, que acabou sendo
recheada de surpresas para a minha vida, e então não tive tempo. Fui muleque.
Para não deixar sem nada no envio, este é o código da classe da GMM que implementei. A alto nível
era para ela estar completa, porém o scipy me diz que há um bug na matriz de covariância a partir
da segunda iteração do fit que não soube como resolver. E este foi o motivo da minha derrota
(além, óbvio, da minha burrice).

- seu querido bolsista Diego
"""


import numpy as np
from scipy.stats import multivariate_normal

class GaussianMixtureModel:
    def __init__(self, num_clusters: int, alpha: np.ndarray, pi:np.ndarray, mu: np.ndarray, 
                 sigma: np.ndarray, m_0: np.ndarray, kapa_0: float, v_0: float, S_0: np.ndarray) -> None:
        
        self.K = num_clusters
        self.alpha = alpha  # (1, K)
        self.pi = pi  # (1, K)
        self.mu = mu  # (K, D)
        self.sigma = sigma  # (K, D, D)
        self.m_0 = m_0  # (1, D)
        self.S_0 = S_0  # (D, D)
        self.kapa_0 = kapa_0
        self.v_0 = v_0
    

    def step_e(self, X: np.ndarray) -> None:
        N = X.shape[0]
        
        for i in range(N):
            for k in range(self.K):
                self.r[i][k] = self.pi[:, k] * multivariate_normal(self.mu[k], self.sigma[k]).pdf(X[i])

        den = np.array([np.sum(self.r, axis=1) for _ in range(self.K)]).T
        self.r = self.r / den


    def step_m(self, X: np.ndarray) -> None:
        N, D = X.shape

        x_bar = np.zeros(self.mu.shape)
        for k in range(self.K):
            self.pi[:,] = (self.alpha[k] - 1 + self.r[:, k].sum()) / (N - self.K + self.alpha.sum())
            x_bar[k] = np.array([self.r[i, k] * X[i] for i in range(N)]).sum(axis=0) / self.r[:, k].sum()
            self.mu[k] = (self.kapa_0 * self.m_0 + x_bar[k] * self.r[:, k].sum()) / (self.kapa_0 + self.r[:, k].sum())
            
            s_aux_1 = np.array([self.r[i, k] * (X[i] - x_bar[k]) * (X[i] - x_bar[k]).T for i in range(N)]).sum(axis=0)
            s_aux_2 = (self.kapa_0 * self.r[:, k].sum())/(self.kapa_0 + self.r[:, k].sum())
            s_aux_3 = (x_bar[k] - self.m_0) * (x_bar[k] - self.m_0).T

            print(s_aux_1.shape)
            print(s_aux_2.shape)
            print(s_aux_3.shape)

            self.sigma[k] = (self.S_0 + s_aux_1 + s_aux_2*s_aux_3)/(self.v_0 + D + 2 + self.r[:, k].sum())


    def fit(self, X: np.ndarray, num_epochs: int) -> None:
        N = X.shape[0]

        self.r = np.zeros((N, self.K))

        for e in range(num_epochs):
            self.step_e(X)
            self.step_m(X)


    def predict(self, X: np.ndarray) -> np.ndarray:
        # should return a matrix of shape (n, K)
        
        pass