import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, learningRate=0.000001,n=1000,l2_reg=True,lambd=0.001):
        self.learningRate = learningRate
        #self.annealing_lr = annealing_lr
        #self.T_lr = T_lr
        self._lambda = lambd
        self.l2_reg = l2_reg
        self.n = n
        self.lossValsTraining = []
        self.lossValsValidation = []
        self.lossValsTest = []
        self.percentCorrectTraining = []
        self.percentCorrectValidation = []
        self.percentCorrectTest = []
        self.weightsLengths = []

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _bias(self, X):
        return np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

    def _loss(self, h, y):
        if self.l2_reg:
            return -(y * np.log(h) + (1 - y) * np.log(1 - h)).mean() + (self._lambda/2)*np.sum(np.square(self.w))
        else: return -(y * np.log(h) + (1 - y) * np.log(1 - h)).mean()

    def _gradient(self, X, h, y):
        if self.l2_reg:
            return (np.dot(X.T, (h - y)) / y.shape[0]) + (self._lambda*self.w)
        else: return np.dot(X.T, (h - y)) / y.shape[0]

    def fit(self, X, y, X_validation, Y_validation, X_test, Y_test):

        #bias trick
        X = self._bias(X)
        X_validation = self._bias(X_validation)
        X_test = self._bias(X_test)

        #create weights matrix
        self.w = np.zeros(X.shape[1])

        for i in range(self.n):
            print("Epoch {0}/{1}".format(i+1,self.n))
            h = self._sigmoid(np.dot(X, self.w))
            #Update step
            self.w -= self.learningRate*self._gradient(X,h,y)

            #Record cost function values
            self.lossValsTraining.append(self._loss(h,y))
            self.lossValsValidation.append(self._loss(self._sigmoid(np.dot(X_validation, self.w)),Y_validation))
            self.lossValsTest.append(self._loss(self._sigmoid(np.dot(X_test, self.w)),Y_test))

            #Record percentage of correctly predicted images
            Y_hat = self.predict(X)
            self.percentCorrectTraining.append(self.score(Y_hat,y))
            Y_hat = self.predict(X_validation)
            self.percentCorrectValidation.append(self.score(Y_hat,Y_validation))
            Y_hat = self.predict(X_test)
            self.percentCorrectTest.append(self.score(Y_hat,Y_test))

            #Record weights length
            self.weightsLengths.append(np.sum(np.square(self.w)))

            #Early stopping
            if len(self.lossValsValidation) > 3:
                if self.lossValsValidation[i-2] <  self.lossValsValidation[i-1] <  self.lossValsValidation[i]:
                    return


    def predict(self, X):
        #bias trick
        #X = self._bias(X)
        #return probability
        return self._sigmoid(np.dot(X, self.w)).round()

    def score(self,Y_hat,Y):
        return 100*np.sum(Y_hat == Y)/np.size(Y_hat)

    def plotWeightsAsImage(self):
        plt.imshow(np.reshape(self.w[1:], (28, 28)), cmap='gray')



