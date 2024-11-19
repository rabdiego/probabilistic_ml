import numpy as np

class PolynomialFeatures:
    def __init__(self, degree: int) -> None:
        self.d = degree
    

    def transform(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        X_aux = np.hstack((np.array([np.ones(n)]).T, X))
        for i in range(2, self.d+1):
            X_aux = np.hstack((X_aux, np.power(X, i)))
        
        return X_aux

