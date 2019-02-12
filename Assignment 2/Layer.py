import numpy as np
import utils

class Layer:
    def __init__(self,num_neurons=64,num_input=0,activation_func=utils.sigmoid, activation_func_der=utils.sigmoid_der):
        self._num_neurons = num_neurons
        self._num_input = num_input
        self._activation_func = activation_func
        self._activation_func_der = activation_func_der

        self.delta = 0
        self.dw = 0
        self.a = 0


        self.w = np.random.uniform(-1, 1, (num_neurons, self._num_input))

    def activation(self,z):
        return self._activation_func(z)

    def activation_der(self,a):
        return self._activation_func_der(a)
