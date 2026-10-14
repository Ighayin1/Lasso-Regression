import numpy as np

class Lasso_Regression:
    def __init__(self, learning_rate, iterations, lambda_param):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.lambda_param = lambda_param
        
    def fit(self, X, Y):
        self.m, self.n = X.shape
        self.w = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y
        
        for _ in range(self.iterations):
            self.update_weights()
        return self
        
    def update_weights(self):
        Y_prediction = self.predict(self.X)
        db = (-2 * np.sum(self.Y - Y_prediction))/self.m
        dw = (-2 * (self.X.T).dot(self.Y - Y_prediction) + self.lambda_param * np.sign(self.w))/self.m
        #np.sign(self.w) takes care or the condition for w > 0 & w <= 0
            
        #updating weights
        self.w = self.w - self.learning_rate * dw
        self.b = self.b - self.learning_rate * db
        
    def predict(self, X):
        return X.dot(self.w) + self.b