import mnist
import matplotlib.pyplot as plt
import numpy as np
from LogisticRegression import *

mnist.init()
X_train, Y_train, X_test, Y_test = mnist.load()



#Slice the training set
Y_train = Y_train[:200]
X_train = X_train[:200,:]

#Delete all images that we dn't want to classify
indices = np.where(np.logical_and(Y_train != 2,Y_train !=3))
Y_train = np.delete(Y_train, indices)
X_train = np.delete(X_train, indices, axis=0)
Y_train[Y_train == 2] = 1
Y_train[Y_train == 3] = 0


indices = np.where(np.logical_and(Y_test != 2,Y_test !=3))
Y_test = np.delete(Y_test, indices)
X_test = np.delete(X_test, indices, axis=0)
Y_test[Y_test == 2] = 1
Y_test[Y_test == 3] = 0



np.true_divide(X_train, 255)
np.true_divide(X_test, 255)

model = LogisticRegression(lr=0.001, num_iter=100)
model.fit(X_train, Y_train)

print('X_test ', X_test.shape)
correct = 0
y_hat = model.predict(X_test)
for i in range(0,len(y_hat)):
    if Y_test[i]==y_hat[i]:
        correct+=1

print(correct/len(X_test))
#print(model.predict(X_test[-1],0.5))
#plt.imshow(X_train[0],cmap='gray')
#plt.show()





