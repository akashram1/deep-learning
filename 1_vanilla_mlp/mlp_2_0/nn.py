import numpy as np
import pandas as pd
from project import nn_building_blocks as nbb
from project import data_retriever as dr


class NN(object):
    def __init__(self):
        self.architecture = []

    def add_layer(self, layer):
        assert 'layer should be of Type NNLayer', layer in nbb.NNLayer
        self.architecture.append(layer)

    def add_activation(self, activation_function):
        self.architecture.append(activation_function)

    def train(self, x_train, y_train, x_valid, y_valid, mini_batch_size, learning_rate):
        iterations_per_epoch = int(x_train.shape[0] / mini_batch_size)

        # Early stopping parameters
        global j
        global prev_val_loss
        prev_val_loss = float('inf')
        j = 0

        # Stopping criteria
        stopping_criteria = False

        # Initialize parameters
        self.initialize_params(input_dimension=x_train.shape[1])


        # while stopping_criteria:

    def initialize_params(self, input_dimension):


if __name__ == '__main__':
    # Get data
    x_train, y_train, x_valid, y_valid, x_test, y_test = dr.DataRetriever.get_data(data_folder='data')
    nn_arch = NN()

    # Train with random weight init
    nn_arch.add_layer(layer=nbb.FullyConnected(num_units=100, input_dimension=x_train.shape[0]))
    nn_arch.add_activation(activation_function=nbb.ActivationFunction.sigmoid)
    nn_arch.add_layer(layer=nbb.FullyConnected(num_units=10))
    nn_arch.add_activation(activation_function=nbb.ActivationFunction.softmax)
    [avg_trn_error, avg_vldn_error] = nn_arch.train(x_train=x_train, y_train=y_train,
                                                    x_valid=x_valid, y_valid=y_valid,
                                                    mini_batch_size=32, learning_rate=0.1, )
    classfication_accuracy = nn_arch.test(x_test=x_test, y_test=y_test)

    # Train with pre-trained weight init
    w = np.zeros((x_train.shape[1], 100))
    nn_arch.add_layer(layer=nbb.FullyConnected(num_units=100, input_dimension=x_train.shape[0], w=w))
    nn_arch.add_activation(activation_function=nbb.ActivationFunction.sigmoid)
    nn_arch.add_layer(layer=nbb.FullyConnected(num_units=10))
    nn_arch.add_activation(activation_function=nbb.ActivationFunction.softmax)
    [avg_trn_error, avg_vldn_error] = nn_arch.train(x_train=x_train, y_train=y_train,
                                                    x_valid=x_valid, y_valid=y_valid,
                                                    mini_batch_size=32, learning_rate=0.1, )
    classfication_accuracy = nn_arch.test(x_test=x_test, y_test=y_test)