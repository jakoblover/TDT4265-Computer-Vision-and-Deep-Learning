import numpy as np
import matplotlib.pyplot as plt


class SoftmaxRegression:

    def __init__(self, learningRate=0.000001, n=10000, lambd=0.01):
        self.n = n
        self.learningRate = learningRate
        self._lambda = lambd
        self.lossValsTraining = []
        self.lossValsValidation = []
        self.lossValsTest = []
        self.percentCorrectTraining = []
        self.percentCorrectValidation = []
        self.percentCorrectTest = []

    def _vectorToOneHot(self,vector):
        return np.eye(10)[vector]

    def _oneHotToVector(self,matrix):
        return np.where(matrix==1)

    def _cost(self, Y, h):
        return (-1/Y.shape[0])*(Y * np.log(h)).mean() + (self._lambda / 2) * np.sum(np.square(self.w))

    def _softmax(self, z):
        z -= np.max(z)
        return (np.exp(z).T / np.sum(np.exp(z), axis=1)).T

    def _gradient(self,X, Y, h):
        return (-1 / X.shape[0]) * np.dot(X.T, (Y - h)) + self._lambda * self.w

    def fit(self, X_train, Y_train, X_validation, Y_validation, X_test, Y_test):
        Y_train = self._vectorToOneHot(Y_train)
        Y_validation = self._vectorToOneHot(Y_validation)
        Y_test = self._vectorToOneHot(Y_test)
        self.w = np.zeros([X_train.shape[1],10])

        for i in range(0, self.n):
            print('Epoch {0}'.format(i))
            h = self._softmax(np.dot(X_train, self.w))
            self.w -= self.learningRate * self._gradient(X_train, Y_train, h)

            # Record cost function values
            self.lossValsTraining.append(self._cost(Y_train,h))
            self.lossValsValidation.append(self._cost(Y_validation, self._softmax(np.dot(X_validation, self.w))))
            self.lossValsTest.append(self._cost(Y_test, self._softmax(np.dot(X_test, self.w))))

            # Record percentage of correctly predicted images

            self.percentCorrectTraining.append(self.accuracy(X_train, self._oneHotToVector(Y_train)))
            self.percentCorrectValidation.append(self.accuracy(X_validation, self._oneHotToVector(Y_validation)))
            self.percentCorrectTest.append(self.accuracy(X_test, self._oneHotToVector(Y_test)))

            #Early stopping
            if len(self.lossValsValidation) > 3:
                if self.lossValsValidation[i - 2] < self.lossValsValidation[i - 1] < self.lossValsValidation[i]:
                    return

    def predict(self, X):
        return np.argmax(self._softmax(np.dot(X, self.w)), axis=1)

    def accuracy(self, X, Y):
        Y_hat = self.predict(X)
        return np.sum(Y_hat == Y) / np.size(Y_hat)





