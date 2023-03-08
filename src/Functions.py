import numpy as np
from abc import ABC

'''
Define activation functions interface
'''

class ActivationFunction(ABC): 
    @staticmethod
    def f(x) -> float:
        pass

    @staticmethod
    def d(x) -> float:
        pass

'''
Define sigmoid
'''
class Sigmoid(ActivationFunction):
    @staticmethod
    def f(x) -> float:
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def d(x) -> float:
        return Sigmoid.f(x) * (1 - Sigmoid.f(x))
