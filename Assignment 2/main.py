import mnist
import utils
import matplotlib.pyplot as plt
import numpy as np
from NeuralNetwork import *
from Layer import *

#mnist.init()
X_train, Y_train, X_test, Y_test = mnist.load()

# Pre-process data
X_train = X_train[0:40000]
Y_train = Y_train[0:40000]

X_train, X_test = (X_train/127.5)-1, (X_test/127.5)-1
X_train = utils.bias_trick(X_train)
X_test = utils.bias_trick(X_test)
Y_train, Y_test = utils.onehot_encode(Y_train), utils.onehot_encode(Y_test)

X_train, Y_train, X_val, Y_val = utils.train_val_split(X_train, Y_train, 0.1)



#Add Layers
hidden_layer = Layer(num_input=X_train.shape[1], num_neurons=64)
output_layer = Layer(num_input=64, num_neurons=Y_train.shape[1], activation_func=utils.softmax)

#Create network
model = NeuralNetwork(max_epochs=20,learning_rate=1,should_gradient_check=False,batch_size=128)

model.addLayer(hidden_layer)
model.addLayer(output_layer)

w = model.fit(X_train,Y_train,X_val,Y_val,X_test,Y_test)

#print(model.forward(X_test))





plt.plot(model.TRAIN_LOSS, label="Training loss")
plt.plot(model.TEST_LOSS, label="Testing loss")
plt.plot(model.VAL_LOSS, label="Validation loss")
plt.legend()
#plt.ylim([0, 0.05])
plt.show()

plt.clf()
plt.plot(model.TRAIN_ACC, label="Training accuracy")
plt.plot(model.TEST_ACC, label="Testing accuracy")
plt.plot(model.VAL_ACC, label="Validation accuracy")
plt.ylim([0.8, 1.0])
plt.legend()
plt.show()

plt.clf()

'''
w = w[:, :-1]  # Remove bias
w = w.reshape(10, 28, 28)
w = np.concatenate(w, axis=0)
plt.imshow(w, cmap="gray")
plt.show()
'''
