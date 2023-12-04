import numpy as np

class LogisticRegression:
    theta: np.ndarray[float]
    learning_rate: float
    num_iterations: int

    def __init__(self, learning_rate: float = 0.01, num_iter: int = 100000, 
                 fit_intercept: bool = False):
        self.lr = learning_rate
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
    
    def _add_intercept(self, X: np.ndarray):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def _sigmoid(self, z: np.ndarray):
        return 1 / (1 + np.exp(-z))
    
    def _loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def train(self, X: list, Y: list):
        X = np.array(X)
        Y = np.array(Y)
        if self.fit_intercept:
            X = self._add_intercept(X)
        
        self.theta = np.zeros(X.shape[1])
        
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self._sigmoid(z)
            gradient = np.dot(X.T, (h - Y)) / Y.size
            self.theta -= self.lr * gradient
            
            if(self.verbose == True and i % 10000 == 0):
                z = np.dot(X, self.theta)
                h = self._sigmoid(z)
                print(f'loss: {self._loss(h, Y)} \t')
    
    def _predict_prob(self, X):
        if self.fit_intercept:
            X = self._add_intercept(X)
    
        return self._sigmoid(np.dot(X, self.theta))
    
    def predict(self, X, threshold: float = 0.5) -> tuple[float, bool]:
        if(not isinstance(X, np.ndarray)):
            X = np.array(X)
        probability = self._predict_prob(X)
        prediction = probability >= threshold
        return (probability, prediction)