import mnist
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from LogisticRegression import *
from SoftmaxRegression import *
import utils

#mnist.init()
X_train, Y_train, X_test, Y_test = mnist.load()

#Slice the training set
X_validation = X_train[0:2000]
Y_validation = Y_train[0:2000]
X_train = X_train[2000:20000,:]
Y_train = Y_train[2000:20000]
X_test = X_test[-2000:,:]
Y_test = Y_test[-2000:]

#Normalize (We realized this after we finished the assignment, but still included the function for reference)
#X_train,X_validation,X_test = utils.normalize(X_train,X_validation,X_test)





#Example of softmax
model = SoftmaxRegression(learningRate=0.000001, n=100, lambd=0.01)
model.fit(X_train, Y_train, X_validation, Y_validation, X_test, Y_test)
utils.plotWeightsAndAverageDigits(X_train, Y_train, model.w)
utils.plotCostAndAccuracy(model.lossValsTraining,model.lossValsValidation,model.lossValsTest,model.percentCorrectTraining,model.percentCorrectValidation,model.percentCorrectTest)



#Must prepare data for binary classification
X_train,Y_train,X_validation,Y_validation,X_test,Y_test = utils.returnSetsBinaryClassification(X_train,Y_train,X_validation,Y_validation,X_test,Y_test,2,3)

#Example of logistic regression
model = LogisticRegression(learningRate=0.000001, n=500, l2_reg=True, lambd=0.01)
model.fit(X_train, Y_train, X_validation, Y_validation, X_test, Y_test)


