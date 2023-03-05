import numpy as np
from abc import ABC

'''
Define activation functions interface
'''

class ActivationFunction(ABC): 
    @staticmethod
    def f(x):
        pass

    @staticmethod
    def d(x):
        pass

'''
Define sigmoid
'''
class Sigmoid(ActivationFunction):
    @staticmethod
    def f(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def d(x):
        return f(x) * (1 - f(x))
