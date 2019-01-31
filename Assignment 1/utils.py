import numpy as np
import matplotlib.pyplot as plt

def _plotAverageDigit(X, Y, digit):
    num = X[Y == digit].mean(axis=0)
    plt.imshow(np.reshape(num, (28, 28)), cmap='gray')

def _plotWeightsDigit(w,digit):
    plt.imshow(np.reshape(w[:, digit], (28, 28)), cmap='gray')

def returnSetsBinaryClassification(X_train,Y_train,X_validation,Y_validation,X_test,Y_test,digit1, digit2):
    #For Logistic regression - Delete images, and adjust labels
    indices = np.where(np.logical_and(Y_train != digit1,Y_train !=digit2))
    Y_train = np.delete(Y_train, indices)
    X_train = np.delete(X_train, indices, axis=0)
    Y_train[Y_train == digit1] = 1
    Y_train[Y_train == digit2] = 0

    #Delete images in test set
    indices = np.where(np.logical_and(Y_test != digit1,Y_test !=digit2))
    Y_test = np.delete(Y_test, indices)
    X_test = np.delete(X_test, indices, axis=0)
    Y_test[Y_test == digit1] = 1
    Y_test[Y_test == digit2] = 0

    #Delete images in training set
    indices = np.where(np.logical_and(Y_validation != digit1,Y_validation !=digit2))
    Y_validation = np.delete(Y_validation, indices)
    X_validation = np.delete(X_validation, indices, axis=0)
    Y_validation[Y_validation == digit1] = 1
    Y_validation[Y_validation == digit2] = 0

    return X_train,Y_train,X_validation,Y_validation,X_test,Y_test

def normalize(X_train,X_validation,X_test):
    X_train = np.true_divide(X_train, 255)
    X_validation = np.true_divide(X_validation, 255)
    X_test = np.true_divide(X_test, 255)
    return X_train, X_validation, X_test

def plotWeightsAndAverageDigits(X,Y,w):
    fig = plt.figure()
    for i in range(5):
        a = fig.add_subplot(4, 5, i + 1)
        _plotAverageDigit(X,Y,i)
    for i in range(5):
        a = fig.add_subplot(4, 5, i + 6)
        _plotWeightsDigit(w,i)
    for i in range(5,10):
        a = fig.add_subplot(4, 5, i + 6)
        _plotAverageDigit(X,Y,i)
    for i in range(5,10):
        a = fig.add_subplot(4, 5, i + 11)
        _plotWeightsDigit(w,i)
    plt.show()

def plotWeights(w):
    plt.imshow(np.reshape(w[1:], (28, 28)), cmap='gray')

def plotCostAndAccuracy(lossValsTraining, lossValsValidation, lossValsTest, percentCorrectTraining, percentCorrectValidation, percentCorrectTest):
    plt.subplot(2, 1, 1)
    plt.plot(lossValsTraining)
    plt.plot(lossValsValidation)
    plt.plot(lossValsTest)
    plt.ylabel('Cost')
    plt.xlabel('Iteration')
    plt.legend([r'Training set', r'Validation set', r'Test set'])

    plt.subplot(2, 1, 2)
    plt.plot(percentCorrectTraining)
    plt.plot(percentCorrectValidation)
    plt.plot(percentCorrectTest)
    plt.ylabel('% Correct')
    plt.xlabel('Iteration')
    plt.legend([r'Training set', r'Validation set', r'Test set'])

    plt.show()
