import numpy as np


class SoftmaxRegression:

    def __init__(self, learningRate=0.01, n=100000):
        pass

    def _softmaxCost(self, t, Y):
        return -np.sum(np.sum(t*np.log(Y),Y.shape[1]), Y.shape[0])

    def _netOutput(self,W):
        pass

