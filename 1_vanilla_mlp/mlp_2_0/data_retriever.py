import numpy as np

class DataRetriever(object):

    @staticmethod
    def get_data(data_folder):
        train_combined = np.loadtxt(data_folder + '/' + 'digitstrain.txt', delimiter=',')
        valid_combined = np.loadtxt(data_folder + '/' + 'digitsvalid.txt', delimiter=',')
        test_combined = np.loadtxt(data_folder + '/' + 'digitstest.txt', delimiter=',')
        x_train = train_combined[:, :-1]
        y_train = train_combined[:, -1]
        x_valid = valid_combined[:, :-1]
        y_valid = valid_combined[:, -1]
        x_test = test_combined[:, :-1]
        y_test = test_combined[:, -1]
        return x_train, y_train, x_valid, y_valid, x_test, y_test
