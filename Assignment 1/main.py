import mnist
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from LogisticRegression import *
from SoftmaxRegression import *

def plotAverageDigit(X, Y, digit):
    num = X_train[Y_train == digit].mean(axis=0)
    plt.imshow(np.reshape(num, (28, 28)), cmap='gray')


#mnist.init()
X_train, Y_train, X_test, Y_test = mnist.load()



#Slice the training set
X_validation = X_train[0:2000]
Y_validation = Y_train[0:2000]
X_train = X_train[2000:20000,:]
Y_train = Y_train[2000:20000]
X_test = X_test[-2000:,:]
Y_test = Y_test[-2000:]



# #For Logistic regression - Delete images, and adjust labels
# indices = np.where(np.logical_and(Y_train != 2,Y_train !=3))
# Y_train = np.delete(Y_train, indices)
# X_train = np.delete(X_train, indices, axis=0)
# Y_train[Y_train == 2] = 1
# Y_train[Y_train == 3] = 0
#
# #Delete images in test set
# indices = np.where(np.logical_and(Y_test != 2,Y_test !=3))
# Y_test = np.delete(Y_test, indices)
# X_test = np.delete(X_test, indices, axis=0)
# Y_test[Y_test == 2] = 1
# Y_test[Y_test == 3] = 0
#
# #Delete images in training set
# indices = np.where(np.logical_and(Y_validation != 2,Y_validation !=3))
# Y_validation = np.delete(Y_validation, indices)
# X_validation = np.delete(X_validation, indices, axis=0)
# Y_validation[Y_validation == 2] = 1
# Y_validation[Y_validation == 3] = 0

#Normalize
np.true_divide(X_train,255)
np.true_divide(X_test,255)
np.true_divide(X_validation,255)

model = SoftmaxRegression(learningRate=0.000001, n=1000, lambd=0.01)
model.fit(X_train, Y_train, X_validation, Y_validation, X_test, Y_test)

#For softmax - Plot of cost and percentage together
plt.subplot(2, 1, 1)
model.plotCost()
plt.subplot(2, 1, 2)
model.plotPercentageCorrect()
plt.show()


#For softmax - Plot of weights together with average of digits
fig = plt.figure()
for i in range(5):
    a = fig.add_subplot(4, 5, i + 1)
    plotAverageDigit(X_train,Y_train,i)
for i in range(5):
    a = fig.add_subplot(4, 5, i + 6)
    model.plotWeightVisualization(i)
for i in range(5,10):
    a = fig.add_subplot(4, 5, i + 6)
    plotAverageDigit(X_train,Y_train,i)
for i in range(5,10):
    a = fig.add_subplot(4, 5, i + 11)
    model.plotWeightVisualization(i)
plt.show()




# model = LogisticRegression(learningRate=0.000001, n=15000, l2_reg=True, lambd=0.01)
# model.fit(X_train, Y_train, X_validation, Y_validation, X_test, Y_test)
# plt.plot(model.weightsLengths)
#
#
# model = LogisticRegression(learningRate=0.000001, n=15000, l2_reg=True, lambd=0.001)
# model.fit(X_train, Y_train, X_validation, Y_validation, X_test, Y_test)
# plt.plot(model.weightsLengths)
#
#
# model = LogisticRegression(learningRate=0.000001, n=15000, l2_reg=True, lambd=0.0001)
# model.fit(X_train, Y_train, X_validation, Y_validation, X_test, Y_test)
# plt.plot(model.weightsLengths)
#
#
# plt.ylabel('Length of weight vector')
# plt.xlabel('Iteration')
# plt.legend([r'$\lambda = 0.01$', r'$\lambda = 0.001$', r'$\lambda = 0.0001$'])
# plt.show()

