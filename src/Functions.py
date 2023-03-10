import numpy as np
from abc import ABC

'''
Define activation functions interface
'''

class ActivationFunction(ABC): 
    @staticmethod
    def f(x) -> np.ndarray:
        pass

    @staticmethod
    def d(x) -> np.ndarray:
        pass

    '''@staticmethod
    def f2(x, i, j) -> float:
        pass

    @staticmethod
    def d2(x, i, j) -> float:
        pass'''

'''
Define sigmoid
'''
class Sigmoid(ActivationFunction):
    @staticmethod
    def f(x) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def d(x) -> np.ndarray:
        return Sigmoid.f(x) * (1 - Sigmoid.f(x))
'''
Define softmax
'''
class SoftMax(ActivationFunction):
    @staticmethod
    def f(x) -> np.ndarray:
        #print(x)
        expZ = np.exp(x - np.max(x))
        return expZ / expZ.sum(axis=0, keepdims=True)

    #@staticmethod
    #def f2(x) -> float:
    #    return np.exp(x[i]) / np.sum(np.exp(x))

    @staticmethod
    def d(x:np.ndarray) -> np.ndarray:
        I = np.eye(x.shape[0])

        return SoftMax.f(x) * (I - SoftMax.f(x).T)
