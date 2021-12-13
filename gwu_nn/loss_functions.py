import numpy as np
from gwu_nn.activation_functions import SoftmaxActivation
from abc import ABC, abstractmethod


class LossFunction(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def loss(cls, y_true, y_pred):
        pass

    @abstractmethod
    def loss_partial_derivative(cls, y_true, y_pred):
        pass


class MSE(LossFunction):

    @classmethod
    def loss(cls, y_true, y_pred):
        return np.mean(np.power(y_true - y_pred, 2))

    @classmethod
    def loss_partial_derivative(cls, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_pred.size

class LogLoss(LossFunction):

    @classmethod
    def loss(cls, y_true, y_pred):
        return np.mean(-np.log(y_pred)*y_true + -np.log(1-y_pred)*(1-y_true))

    @classmethod
    def loss_partial_derivative(cls, y_true, y_pred):
        return -np.sum(y_true - y_pred)


class CrossEntropy(LossFunction):

    @classmethod
    def loss(cls, y_true, y_pred):
        return -np.mean(y_true*np.log(y_pred))

    @classmethod
    def loss_partial_derivative(cls, y_true, y_pred):
        m = y_true.shape[0]
        grad = SoftmaxActivation.activation(y_pred)
        grad[range(m), y_true] -= 1
        grad = grad / m
        return grad

class MultiClassCrossEntropy(LossFunction):

    @classmethod
    def loss(cls, y_true, y_pred):
        # this looks correct
        #print("multiclass loss func")
        #print("mclf " + str(y_true.shape))
        #print("mclf " + str(y_pred.shape))

        # compute softmax on y_pred
        epsilon = 1e-9
        exps = np.exp(y_pred - np.max(y_pred))
        softmax = exps / np.sum(exps)
        softmax = np.clip(softmax, epsilon, 1 - epsilon) # added clipping to prevent infinity 

        softmax_cross_entropy_loss = -1.0 * y_true * np.log(softmax) - (1.0 - y_true) * np.log(1 - softmax)
        #print(softmax_cross_entropy_loss)
        sm_sum = np.sum(softmax_cross_entropy_loss)
        #print("sm_sum " + str(sm_sum))
        return sm_sum

    @classmethod
    def loss_partial_derivative(cls, y_true, y_pred):
        # compute softmax on y_pred
        epsilon = 1e-9
        exps = np.exp(y_pred - np.max(y_pred))
        softmax = exps / np.sum(exps)
        y_pred = np.clip(softmax, epsilon, 1 - epsilon) # added clipping to prevent infinity 

        #print("is this loss prime1?")
        #print("mclf1 " + str(y_true.shape))
        #print("mclf1 " + str(y_pred.shape))

        loss = y_pred - y_true
        #loss = loss.reshape(-1,1) # reshape???

        return loss#y_pred - y_true