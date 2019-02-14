import numpy as np
import matplotlib.pyplot as plt
import mnist
import tqdm
import utils

class NeuralNetwork:
    def __init__(self,max_epochs=40, learning_rate=0.5, should_gradient_check=False, batch_size = 128, momentum = 0.9):
        self._batch_size = batch_size
        self._max_epochs = max_epochs
        self._learning_rate = learning_rate
        self._should_gradient_check = should_gradient_check
        self._momentum = momentum

        self.layers = []

        # Tracking variables
        self.TRAIN_LOSS = []
        self.TEST_LOSS = []
        self.VAL_LOSS = []
        self.TRAIN_ACC = []
        self.TEST_ACC = []
        self.VAL_ACC = []

    def addLayer(self, layer):
        self.layers.append(layer)

    def forward(self, X_in):
        """
        Performs a forward pass through the network
        --
        X: shape: [batch_size, num_features(784+1)] numpy vector. Input batch of images
        w: shape: [num_classes, num_features] numpy vector. Weight from input->output
        --
        Returns: [batch_size, num_classes] numpy vector
        """

        #Iterate through the network and update activation
        a = 0
        for i in range(0,len(self.layers)):
            if i == 0:
                z = X_in.dot(self.layers[i].w.T)
                a = self.layers[i].activation(z)
            else:
                z = a.dot(self.layers[i].w.T)
                a = self.layers[i].activation(z)

            self.layers[i].a = a
            self.layers[i].z = z

        return a

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


                loss1 = self._cross_entropy_loss1(X, targets, self.layers[0].w, new_weight1)
                loss2 = self._cross_entropy_loss1(X, targets, self.layers[0].w, new_weight2)


                dw[k,j] = (loss1 - loss2) / (2*epsilon)
        maximum_abosulte_difference = abs(computed_gradient-dw).max()
        print(maximum_abosulte_difference)
        assert maximum_abosulte_difference <= epsilon**2, "Absolute error was: {}".format(maximum_abosulte_difference)


    def _calculate_accuracy(self, X, targets):
        """
        Calculated the accuracy of the network.
        ---
        X: shape: [batch_size, num_features(784+1)] numpy vector. Input batch of images
        targets: shape: [batch_size, num_classes] numpy vector. Targets/label of images
        w: shape: [num_classes, num_features] numpy vector. Weight from input->output
        --
        Returns float
        """
        output = self.forward(X)
        predictions = output.argmax(axis=1)
        targets = targets.argmax(axis=1)
        return (predictions == targets).mean()


    def _cross_entropy_loss(self, X, targets):
        """
        Computes the cross entropy loss given the input vector X and the target vector.
        ---
        X: shape: [batch_size, num_features(784+1)] numpy vector. Input batch of images
        targets: shape: [batch_size, num_classes] numpy vector. Targets/label of images
        w: shape: [num_classes, num_features] numpy vector. Weight from input->output
        --
        Returns float
        """
        output = self.forward(X)
        assert output.shape == targets.shape
        log_y = np.log(output)
        cross_entropy = -targets * log_y
        return cross_entropy.mean()

    def _gradient_descent(self, X, targets):
        """
        Performs gradient descents for all weights in the network.
        ---
        X: shape: [batch_size, num_features(784+1)] numpy vector. Input batch of images
        targets: shape: [batch_size, num_classes] numpy vector. Targets/label of images
        w: shape: [num_classes, num_features] numpy vector. Weight from input->output
        --
        """

        # Since we are taking the .mean() of our loss, we get the normalization factor to be 1/(N*C)
        # If you take loss.sum(), the normalization factor is 1.
        # The normalization factor is identical for all weights in the network (For multi-layer neural-networks as well.)
        normalization_factor = X.shape[0] * targets.shape[1] # batch_size * num_classes
        outputs = self.forward(X)

        #Go backwards in our network, and calculate deltas
        for i in range(len(self.layers)-1,-1,-1):
            if i == len(self.layers)-1:
                self.layers[i].delta = - (targets - outputs)
            else:
                self.layers[i].delta = np.multiply(self.layers[i].activation_der(self.layers[i].z),np.dot(self.layers[i+1].delta,self.layers[i+1].w))

        #Go forwards in our network and update our gradients
        for i in range(0,len(self.layers)):
            self.layers[i].prev_dw = self.layers[i].dw
            if i == 0:
                self.layers[i].dw = self.layers[i].delta.T.dot(X) / normalization_factor
            else:
                self.layers[i].dw = self.layers[i].delta.T.dot(self.layers[i-1].a) / normalization_factor
            assert self.layers[i].dw.shape == self.layers[i].w.shape, "dw shape was: {}. Expected: {}".format(self.layers[i].dw.shape, self.layers[i].w.shape)




        #if self._should_gradient_check:
        #    self._check_gradient(X, targets, self.layers[1].w, 1e-2,  self.layers[1].dw)


        #Update weights based on new gradient
        for i in range(0,len(self.layers)):
            self.layers[i].w -= self._learning_rate*self.layers[i].dw + self._momentum*self.layers[i].prev_dw


    def fit(self, X_train, Y_train, X_val, Y_val, X_test, Y_test):
        # Tracking variables
        self.TRAIN_LOSS = []
        self.TEST_LOSS = []
        self.VAL_LOSS = []
        self.TRAIN_ACC = []
        self.TEST_ACC = []
        self.VAL_ACC = []
        num_batches = X_train.shape[0] // self._batch_size
        check_step = num_batches // 10


        for e in range(self._max_epochs):  # Epochs
            for i in tqdm.trange(num_batches):
                X_batch = X_train[i*self._batch_size:(i+1)*self._batch_size]
                Y_batch = Y_train[i*self._batch_size:(i+1)*self._batch_size]

                self._gradient_descent(X_batch, Y_batch)

                if i % check_step == 0:
                    # Loss
                    self.TRAIN_LOSS.append(self._cross_entropy_loss(X_train, Y_train))
                    self.TEST_LOSS.append(self._cross_entropy_loss(X_test, Y_test))
                    self.VAL_LOSS.append(self._cross_entropy_loss(X_val, Y_val))

                    self.TRAIN_ACC.append(self._calculate_accuracy(X_train, Y_train))
                    self.VAL_ACC.append(self._calculate_accuracy(X_val, Y_val))
                    self.TEST_ACC.append(self._calculate_accuracy(X_test, Y_test))
                    if self._should_early_stop(self.VAL_LOSS):
                        print(self.VAL_LOSS[-4:])
                        print("early stopping.")
                        return

            #Shuffle training data after every epich
            X_train, Y_train = utils.shuffle(X_train,Y_train)
