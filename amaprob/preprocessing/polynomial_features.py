import numpy as np
from sklearn.preprocessing import StandardScaler

class PolynomialFeatures:
    def __init__(self, degree: int) -> None:
        self.d = degree
    

    def transform(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        X_aux = X
        for i in range(2, self.d+1):
            X_aux = np.hstack((X_aux, np.power(X, i)))
        
        X_aux = StandardScaler().fit_transform(X_aux)
        X_aux = np.hstack((np.array([np.ones(n)]).T, X_aux))

        return X_aux

