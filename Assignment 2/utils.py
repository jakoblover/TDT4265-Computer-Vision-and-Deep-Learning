import numpy as np

def train_val_split(X, Y, val_percentage):
    """
    Selects samples from the dataset randomly to be in the validation set.
    Also, shuffles the train set.
    --
    X: [N, num_features] numpy vector,
    Y: [N, 1] numpy vector
    val_percentage: amount of data to put in validation set
    """
    dataset_size = X.shape[0]
    idx = np.arange(0, dataset_size)
    np.random.shuffle(idx)

    train_size = int(dataset_size*(1-val_percentage))
    idx_train = idx[:train_size]
    idx_val = idx[train_size:]
    X_train, Y_train = X[idx_train], Y[idx_train]
    X_val, Y_val = X[idx_val], Y[idx_val]
    return X_train, Y_train, X_val, Y_val

def shuffle(X, Y):
    dataset_size = X.shape[0]
    idx = np.arange(0, dataset_size)
    np.random.shuffle(idx)

    return X[idx], Y[idx]

def onehot_encode(Y, n_classes=10):
    onehot = np.zeros((Y.shape[0], n_classes))
    onehot[np.arange(0, Y.shape[0]), Y] = 1
    return onehot

def bias_trick(X):
    """
    X: shape[batch_size, num_features(784)]
    --
    Returns [batch_size, num_features+1 ]
    """
    return np.concatenate((X, np.ones((len(X), 1))), axis=1)

def softmax(z):
    """
    Applies the softmax activation function for the vector a.
    --
    a: shape: [batch_size, num_classes]. Activation of the output layer before activation
    --
    Returns: [batch_size, num_classes] numpy vector
    """
    a_exp = np.exp(z)
    return a_exp / a_exp.sum(axis=1, keepdims=True)

def softmax_der(z):
    return

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_der(z):
    return np.multiply(sigmoid(z), 1-sigmoid(z))

def improved_sigmoid(z):
    return 1.7159*np.tanh((2.0/3.0)*z) + 0.001*z

def improved_sigmoid_der(z):
    return 1.7159*(2.0/3.0)*(1/((np.cosh(2.0/3.0*z))**2)) + 0.001

def relu(z):
    return np.maximum(0,z)

def relu_der(z):
    z[z <= 0] = 0
    z[z > 0] = 1
    return z
