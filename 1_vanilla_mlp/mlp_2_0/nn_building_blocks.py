from enum import Enum
import numpy as np


class FullyConnected(object):
    def __init__(self, num_units, input_dimension=None, w=None):
        self.num_units = num_units
        if w is not None:
            self.w = w
        else:
            if input_dimension is not None:
                self.w = get_gaussian_weights(shape=(input_dimension, self.num_units))


class ActivationFunction(Enum):
    none = 0
    sigmoid = 1
    relu = 2
    tanh = 3
    softmax = 4



class NNLayerType(Enum):
    input = 1
    fully_connected = 2
    output = 3


def get_gaussian_weights(shape):
    normal_mean = 0
    normal_stddev = 0.1
    return np.random.normal(loc=normal_mean, scale=normal_stddev, size=shape)
