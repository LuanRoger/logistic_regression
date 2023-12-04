import numpy as np

class LogisticRegression:
    theta: np.ndarray[float]
    learning_rate: float
    num_iterations: int

    def __init__(self, learning_rate: float = 0.01, num_iter: int = 100000):
        self.lr = learning_rate
        self.num_iter = num_iter
    
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def _loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def train(self, X: list, Y: list):
        X = np.array(X)
        Y = np.array(Y)
        
        self.theta = np.zeros(X.shape[1])
        
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self._sigmoid(z)
            gradient = np.dot(X.T, (h - Y)) / Y.size
            self.theta -= self.lr * gradient
    
    def _predict_prob(self, X):
        return self._sigmoid(np.dot(X, self.theta))
    
    def predict(self, X, threshold: float = 0.5) -> tuple[float, bool]:
        if(not isinstance(X, np.ndarray)):
            X = np.array(X)
        probability = self._predict_prob(X)
        prediction = probability >= threshold
        return (probability, prediction)