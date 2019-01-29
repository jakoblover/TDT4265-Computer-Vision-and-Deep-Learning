import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, learningRate=0.000001, n=1000):
        self.learningRate = learningRate
        self.n = n
        self.lossVals = []

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _bias(self, X):
        return np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

    def _loss(self, h, y):
        return -(-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def _gradient(self, X, h, y):
        return np.dot(X.T, (h - y)) / y.shape[0]

    def fit(self, X, y):
        self.lossVals = []
        #bias trick
        X = self._bias(X)
        #create weights matrix
        self.w = np.zeros(X.shape[1])

        for i in range(self.n):
            h = self._sigmoid(np.dot(X, self.w))
            self.w -= self.learningRate*self._gradient(X,h,y)
            self.lossVals.append(self._loss(h,y))

    def predict(self, X):
        #bias trick
        X = self._bias(X)
        #return probability
        return self._sigmoid(np.dot(X, self.w)).round()
