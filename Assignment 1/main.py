import mnist
import time
import matplotlib.pyplot as plt
import numpy as np
from LogisticRegression import *

#start_time = time.time()
#print('Execution time: {0}'.format(time.time()-start_time))

#mnist.init()
X_train, Y_train, X_test, Y_test = mnist.load()



#Slice the training set
X_validation = X_train[0:2000]
Y_validation = Y_train[0:2000]
X_train = X_train[2000:20000,:]
Y_train = Y_train[2000:20000]
X_test = X_test[-2000:,:]
Y_test = Y_test[-2000:]



#Delete images in training set
indices = np.where(np.logical_and(Y_train != 2,Y_train !=3))
Y_train = np.delete(Y_train, indices)
X_train = np.delete(X_train, indices, axis=0)
Y_train[Y_train == 2] = 1
Y_train[Y_train == 3] = 0

#Delete images in test set
indices = np.where(np.logical_and(Y_test != 2,Y_test !=3))
Y_test = np.delete(Y_test, indices)
X_test = np.delete(X_test, indices, axis=0)
Y_test[Y_test == 2] = 1
Y_test[Y_test == 3] = 0

#Delete images in training set
indices = np.where(np.logical_and(Y_validation != 2,Y_validation !=3))
Y_validation = np.delete(Y_validation, indices)
X_validation = np.delete(X_validation, indices, axis=0)
Y_validation[Y_validation == 2] = 1
Y_validation[Y_validation == 3] = 0



#Normalize
np.true_divide(X_train,255)
np.true_divide(X_test,255)
np.true_divide(X_validation,255)


model = LogisticRegression(learningRate=0.000001, n=1500)

model.fit(X_train, Y_train)
plt.plot(model.lossVals)

model.fit(X_validation, Y_validation)
plt.plot(model.lossVals)

model.fit(X_test, Y_test)
plt.plot(model.lossVals)


plt.ylabel('Cost')
plt.xlabel('Iteration')
plt.legend([r'Training set', r'Validation set', r'Test set'])
plt.show()

correct = 0
y_hat = model.predict(X_test)
for i in range(0,len(y_hat)):
    if Y_test[i]==y_hat[i]:
        correct+=1
print(correct/len(X_test))
#print(model.predict(X_test[-1],0.5))
#plt.imshow(X_train[0],cmap='gray')
#plt.show()





