import mnist
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from LogisticRegression import *
from SoftmaxRegression import *

#mnist.init()
X_train, Y_train, X_test, Y_test = mnist.load()



#Slice the training set
X_validation = X_train[0:2000]
Y_validation = Y_train[0:2000]
X_train = X_train[2000:20000,:]
Y_train = Y_train[2000:20000]
X_test = X_test[-2000:,:]
Y_test = Y_test[-2000:]



# #Delete images in training set
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



model = SoftmaxRegression(learningRate=0.000001, n=1500, lambd=0.01)
model.fit(X_train, Y_train, X_validation, Y_validation, X_test, Y_test)

print(model.getAccuracy(X_train,Y_train))
print(model.getAccuracy(X_test,Y_test))

model.plotPercentageCorrect()








# model = LogisticRegression(learningRate=0.000001, n=15000, l2_reg=True, lambd=1)
# model.fit(X_train, Y_train, X_validation, Y_validation, X_test, Y_test)
# plt.plot(model.lossValsValidation)
# model = LogisticRegression(learningRate=0.000001, n=15000, l2_reg=True, lambd=0.001)
# model.fit(X_train, Y_train, X_validation, Y_validation, X_test, Y_test)
# plt.plot(model.lossValsValidation)
# model = LogisticRegression(learningRate=0.000001, n=15000, l2_reg=True, lambd=0.0001)
# model.fit(X_train, Y_train, X_validation, Y_validation, X_test, Y_test)
# plt.plot(model.lossValsValidation)
#
#
# plt.ylabel('Validation Accuracy [%]')
# plt.xlabel('Iteration')
# plt.legend([r'\lambda = 0.01', r'\lambda = 0.001', r'\lambda = 0.0001'])
# plt.show()

#print(model.percentCorrectTest[-1])
#plt.imshow(vectorToImage(model.w[1:]), cmap='gray')
#plt.show()

# plt.subplot(2, 1, 1)
# plt.plot(model.lossValsTraining)
# plt.plot(model.lossValsValidation)
# plt.plot(model.lossValsTest)
# plt.ylabel('Cost')
# plt.xlabel('Iteration')
# plt.legend([r'Training set', r'Validation set', r'Test set'])
#
# plt.subplot(2, 1, 2)
# plt.plot(model.percentCorrectTraining)
# plt.plot(model.percentCorrectValidation)
# plt.plot(model.percentCorrectTest)
# plt.ylabel('% Correct')
# plt.xlabel('Iteration')
# plt.legend([r'Training set', r'Validation set', r'Test set'])
#plt.show()

