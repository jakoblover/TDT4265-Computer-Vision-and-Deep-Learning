import mnist
import utils
from SoftmaxRegression import *

#mnist.init()
X_train, Y_train, X_test, Y_test = mnist.load()

# Pre-process data
X_train, X_test = X_train / 255, X_test / 255
X_train = utils.bias_trick(X_train)
X_test = utils.bias_trick(X_test)
Y_train, Y_test = utils.onehot_encode(Y_train), utils.onehot_encode(Y_test)

X_train, Y_train, X_val, Y_val = utils.train_val_split(X_train, Y_train, 0.1)

model = SoftmaxRegression()
w = model.fit(X_train,Y_train,X_val,Y_val,X_test,Y_test)

plt.plot(model.TRAIN_LOSS, label="Training loss")
plt.plot(model.TEST_LOSS, label="Testing loss")
plt.plot(model.VAL_LOSS, label="Validation loss")
plt.legend()
plt.ylim([0, 0.05])
plt.show()

plt.clf()
plt.plot(model.TRAIN_ACC, label="Training accuracy")
plt.plot(model.TEST_ACC, label="Testing accuracy")
plt.plot(model.VAL_ACC, label="Validation accuracy")
plt.ylim([0.8, 1.0])
plt.legend()
plt.show()

plt.clf()

w = w[:, :-1]  # Remove bias
w = w.reshape(10, 28, 28)
w = np.concatenate(w, axis=0)
plt.imshow(w, cmap="gray")
plt.show()
