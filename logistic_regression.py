import numpy as np

class LogisticRegression:
    theta: np.ndarray[float]
    learning_rate: float
    num_iterations: int

    def __init__(self, learning_rate: float = 0.01, num_iter: int = 100000, 
                 fit_intercept=False, verbose=False):
        self.lr = learning_rate
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
    
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        if self.fit_intercept:
            X = self.__add_intercept(X)
        
        self.theta = np.zeros(X.shape[1])
        
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient
            
            if(self.verbose == True and i % 10000 == 0):
                z = np.dot(X, self.theta)
                h = self.__sigmoid(z)
                print(f'loss: {self.__loss(h, y)} \t')
    
    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
    
        return self.__sigmoid(np.dot(X, self.theta))
    
    def predict(self, X, threshold):
        if(not isinstance(X, np.ndarray)):
            X = np.array(X)
        probability = self.predict_prob(X)
        prediction = probability >= threshold
        return (probability, prediction)
    
    def accuracy(self, X, y):
        if(not isinstance(X, np.ndarray)):
            X = np.array(X)
        if(not isinstance(y, np.ndarray)):
            y = np.array(y)
        (_, predictions) = self.predict(X, 0.5)
        accuracy = (predictions == y).mean()
        return accuracy
    
    def precision(self, X, y):
        if not isinstance(X, (list, np.ndarray)):
            X = np.array(X)
        if not isinstance(y, (list, np.ndarray)):
            y = np.array(y)

        X = np.array(X)
        y = np.array(y)
        (_, predictions) = self.predict(X, 0.5)
        TP = np.sum((predictions == 1) & (y == 1))
        FP = np.sum((predictions == 1) & (y == 0))
        precision = TP / (TP + FP)
        return precision