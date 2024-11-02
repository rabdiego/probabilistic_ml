import numpy as np

class BernoulliNaiveBayes:
    def __init__(self, a: float, b: float) -> None:
        self.prior: dict = {
            'a' : a,
            'b' : b
        }

        self.num_classes: int = None
        self.num_attributes: int = None
        self.thetas: np.ndarray = None
        self.pis: np.ndarray = None  

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        unique, counts = np.unique(y, return_counts=True)
        self.num_classes = unique.shape[0]
        self.num_attributes = X.shape[1]
        num_instances = y.shape[0]

        self.thetas = np.array([
            [(X[y == c][:, d].sum() + self.prior['b'])/(counts[c] + self.prior['a'] + self.prior['b']) for d in range(self.num_attributes)] for c in range(self.num_classes)
        ]).T

        self.pis = np.array([
            (counts[c] + 1)/(num_instances + self.num_classes) for c in range(self.num_classes) 
        ])


    def predict(self, X: np.ndarray) -> np.ndarray:
        prediction = np.array([
            np.argmax(np.log(self.pis) + (np.log(self.thetas)[X[i]==1]).sum(axis=0) + (np.log(1 - self.thetas)[X[i]==0]).sum(axis=0)) for i in range(X.shape[0])
        ])

        return prediction
