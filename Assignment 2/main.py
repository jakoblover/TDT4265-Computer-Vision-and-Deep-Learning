import mnist
import utils
import matplotlib.pyplot as plt
import numpy as np
from NeuralNetwork import *
from Layer import *

#mnist.init()
X_train, Y_train, X_test, Y_test = mnist.load()

# Pre-process data. For computers with less RAM, we must slice the training set
X_train = X_train[0:50000]
Y_train = Y_train[0:50000]

X_train, X_test = (X_train/127.5)-1, (X_test/127.5)-1
X_train = utils.bias_trick(X_train)
X_test = utils.bias_trick(X_test)
Y_train, Y_test = utils.onehot_encode(Y_train), utils.onehot_encode(Y_test)

X_train, Y_train, X_val, Y_val = utils.train_val_split(X_train, Y_train, 0.1)



#Add Layers
hidden_layer = Layer(num_input=X_train.shape[1], num_neurons=64, activation_func=utils.relu, activation_func_der=utils.relu_der)
output_layer = Layer(num_input=64, num_neurons=Y_train.shape[1], activation_func=utils.softmax) #The derivation of the activation function of the output layer is not used

#Create network
model = NeuralNetwork(max_epochs=20,learning_rate=0.01,should_gradient_check=False,batch_size=256, momentum=0.9)
model.addLayer(hidden_layer)
model.addLayer(output_layer)

#Train the network
model.fit(X_train,Y_train,X_val,Y_val,X_test,Y_test)

plt.subplot(2,1,1)
plt.plot(model.TRAIN_LOSS, label="Training loss")
plt.plot(model.TEST_LOSS, label="Testing loss")
plt.plot(model.VAL_LOSS, label="Validation loss")
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.ylim([0, 0.1])

plt.subplot(2,1,2)
plt.plot(model.TRAIN_ACC, label="Training accuracy")
plt.plot(model.TEST_ACC, label="Testing accuracy")
plt.plot(model.VAL_ACC, label="Validation accuracy")
plt.ylim([0.8, 1.0])
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
