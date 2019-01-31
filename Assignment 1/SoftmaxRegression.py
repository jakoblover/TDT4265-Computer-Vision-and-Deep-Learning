import numpy as np
import matplotlib.pyplot as plt


class SoftmaxRegression:

    def __init__(self, learningRate=0.01, n=10000, lambd=0.01):
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

    def softmax(self, z):
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
            h = self.softmax(np.dot(X_train, self.w))
            self.w -= (self.learningRate * self._gradient(X_train, Y_train, h))

            # Record cost function values
            self.lossValsTraining.append(self._cost(Y_train,h))
            self.lossValsValidation.append(self._cost(Y_validation,self.softmax(np.dot(X_validation, self.w))))
            self.lossValsTest.append(self._cost(Y_test,self.softmax(np.dot(X_test, self.w))))

            # Record percentage of correctly predicted images

            self.percentCorrectTraining.append(self.getAccuracy(X_train,self._oneHotToVector(Y_train)))
            self.percentCorrectValidation.append(self.getAccuracy(X_validation,self._oneHotToVector(Y_validation)))
            self.percentCorrectTest.append(self.getAccuracy(X_test,self._oneHotToVector(Y_test)))

            #Early stopping
            if len(self.lossValsValidation) > 3:
                if self.lossValsValidation[i - 2] < self.lossValsValidation[i - 1] < self.lossValsValidation[i]:
                    return

    def getProbsAndPreds(self, someX):
        probs = self.softmax(np.dot(someX, self.w))
        preds = np.argmax(probs, axis=1)
        return probs, preds

    def getAccuracy(self, X, Y):
        prob, preds = self.getProbsAndPreds(X)
        accuracy = np.sum(preds == Y) / (float(len(preds)))
        return accuracy

    def plotPercentageCorrect(self):
        plt.plot(self.percentCorrectTraining)
        plt.plot(self.percentCorrectValidation)
        plt.plot(self.percentCorrectTest)
        plt.ylabel('% Correct')
        plt.xlabel('Iteration')
        plt.legend([r'Training set', r'Validation set', r'Test set'])

    def plotCost(self):
        plt.plot(self.lossValsTraining)
        plt.plot(self.lossValsValidation)
        plt.plot(self.lossValsTest)
        plt.ylabel('Cost')
        plt.xlabel('Iteration')
        plt.legend([r'Training set', r'Validation set', r'Test set'])

    def plotWeightVisualization(self, digit):
    #     n_images = 10
    #     titles = ['Image (%d)' % i for i in range(0, n_images)]
    #     fig = plt.figure()
    #     for i in range(10):
    #         a = fig.add_subplot(4, np.ceil(n_images / float(4)), i + 1)
    #         plt.imshow(np.reshape(self.w[:, i],(28,28)), cmap='gray')
    #     plt.show()
        plt.imshow(np.reshape(self.w[:, digit], (28, 28)), cmap='gray')


