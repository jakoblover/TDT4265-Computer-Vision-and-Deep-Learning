import numpy as np
import matplotlib.pyplot as plt
import mnist
import tqdm

class SoftmaxRegression:

    def __init__(self, batch_size=64, learning_rate=0.5, should_gradient_check=False, max_epochs=20):
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._num_batches = 0 #computed in fit()
        self._check_step = 0 #computed in fit()
        self._should_gradient_check = should_gradient_check
        self._max_epochs = max_epochs

        # Tracking variables
        self.TRAIN_LOSS = []
        self.VAL_LOSS = []
        self.TEST_LOSS = []
        self.TRAIN_ACC = []
        self.VAL_ACC = []
        self.TEST_ACC = []


    def _should_early_stop(self, validation_loss, num_steps=3):
        """
        Returns true if the validation loss increases
        or stays the same for num_steps.
        --
        validation_loss: List of floats
        num_steps: integer
        """
        if len(validation_loss) < num_steps+1:
            return False

        is_increasing = [validation_loss[i] <= validation_loss[i+1] for i in range(-num_steps-1, -1)]
        return sum(is_increasing) == len(is_increasing)

    def _check_gradient(self, X, targets, w, epsilon, computed_gradient):
        """
        Computes the numerical approximation for the gradient of w,
        w.r.t. the input X and target vector targets.
        Asserts that the computed_gradient from backpropagation is
        correct w.r.t. the numerical approximation.
        --
        X: shape: [batch_size, num_features(784+1)]. Input batch of images
        targets: shape: [batch_size, num_classes]. Targets/label of images
        w: shape: [num_classes, num_features]. Weight from input->output
        epsilon: Epsilon for numerical approximation (See assignment)
        computed_gradient: Gradient computed from backpropagation. Same shape as w.
        """
        print("Checking gradient...")
        dw = np.zeros_like(w)
        for k in range(w.shape[0]):
            for j in range(w.shape[1]):
                new_weight1, new_weight2 = np.copy(w), np.copy(w)
                new_weight1[k,j] += epsilon
                new_weight2[k,j] -= epsilon
                loss1 = self._cross_entropy_loss(X, targets, new_weight1)
                loss2 = self._cross_entropy_loss(X, targets, new_weight2)
                dw[k,j] = (loss1 - loss2) / (2*epsilon)
        maximum_absolute_difference = abs(computed_gradient-dw).max()
        assert maximum_absolute_difference <= epsilon**2, "Absolute error was: {}".format(maximum_absolute_difference)


    def _softmax(self, a):
        """
        Applies the softmax activation function for the vector a.
        --
        a: shape: [batch_size, num_classes]. Activation of the output layer before activation
        --
        Returns: [batch_size, num_classes] numpy vector
        """
        a_exp = np.exp(a)
        return a_exp / a_exp.sum(axis=1, keepdims=True)


    def _forward(self, X, w):
        """
        Performs a forward pass through the network
        --
        X: shape: [batch_size, num_features(784+1)] numpy vector. Input batch of images
        w: shape: [num_classes, num_features] numpy vector. Weight from input->output
        --
        Returns: [batch_size, num_classes] numpy vector
        """
        a = X.dot(w.T)
        return self._softmax(a)


    def calculate_accuracy(self, X, targets, w):
        """
        Calculated the accuracy of the network.
        ---
        X: shape: [batch_size, num_features(784+1)] numpy vector. Input batch of images
        targets: shape: [batch_size, num_classes] numpy vector. Targets/label of images
        w: shape: [num_classes, num_features] numpy vector. Weight from input->output
        --
        Returns float
        """
        output = self._forward(X, w)
        predictions = output.argmax(axis=1)
        targets = targets.argmax(axis=1)
        return (predictions == targets).mean()


    def _cross_entropy_loss(self, X, targets, w):
        """
        Computes the cross entropy loss given the input vector X and the target vector.
        ---
        X: shape: [batch_size, num_features(784+1)] numpy vector. Input batch of images
        targets: shape: [batch_size, num_classes] numpy vector. Targets/label of images
        w: shape: [num_classes, num_features] numpy vector. Weight from input->output
        --
        Returns float
        """
        output = self._forward(X, w)
        assert output.shape == targets.shape
        log_y = np.log(output)
        cross_entropy = -targets * log_y
        return cross_entropy.mean()


    def _gradient_descent(self, X, targets, w, learning_rate, should_check_gradient):
        """
        Performs gradient descents for all weights in the network.
        ---
        X: shape: [batch_size, num_features(784+1)] numpy vector. Input batch of images
        targets: shape: [batch_size, num_classes] numpy vector. Targets/label of images
        w: shape: [num_classes, num_features] numpy vector. Weight from input->output
        --
        Returns updated w, with same shape
        """

        # Since we are taking the .mean() of our loss, we get the normalization factor to be 1/(N*C)
        # If you take loss.sum(), the normalization factor is 1.
        # The normalization factor is identical for all weights in the network (For multi-layer neural-networks as well.)
        normalization_factor = X.shape[0] * targets.shape[1] # batch_size * num_classes
        outputs = self._forward(X, w)
        delta_k = - (targets - outputs)

        dw = delta_k.T.dot(X)
        dw = dw / normalization_factor  # Normalize gradient equally as we do with the loss
        assert dw.shape == w.shape, "dw shape was: {}. Expected: {}".format(dw.shape, w.shape)

        if should_check_gradient:
            self._check_gradient(X, targets, w, 1e-2, dw)

        w = w - learning_rate * dw
        return w

    def fit(self, X_train, Y_train, X_val, Y_val, X_test, Y_test):
        self._num_batches = X_train.shape[0] // self._batch_size
        self._check_step = self._num_batches // 10

        w = np.zeros((Y_train.shape[1], X_train.shape[1]))

        for e in range(self._max_epochs):  # Epochs
            for i in tqdm.trange(self._num_batches):
                X_batch = X_train[i*self._batch_size:(i + 1) * self._batch_size]
                Y_batch = Y_train[i*self._batch_size:(i + 1) * self._batch_size]

                w = self._gradient_descent(X_batch,
                                           Y_batch,
                                           w,
                                           self._learning_rate,
                                           self._should_gradient_check)
                if i % self._check_step == 0:
                    # Loss
                    self.TRAIN_LOSS.append(self._cross_entropy_loss(X_train, Y_train, w))
                    self.TEST_LOSS.append(self._cross_entropy_loss(X_test, Y_test, w))
                    self.VAL_LOSS.append(self._cross_entropy_loss(X_val, Y_val, w))

                    self.TRAIN_ACC.append(self.calculate_accuracy(X_train, Y_train, w))
                    self.VAL_ACC.append(self.calculate_accuracy(X_val, Y_val, w))
                    self.TEST_ACC.append(self.calculate_accuracy(X_test, Y_test, w))
                    if self._should_early_stop(self.VAL_LOSS):
                        print(self.VAL_LOSS[-4:])
                        print("early stopping.")
                        return w
        return w









