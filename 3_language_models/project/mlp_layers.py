import numpy as np
from enum import Enum
import math


class Embedding(object):
    def __init__(self, vocab_size, embedding_dim):
        self.v = vocab_size
        self.d = embedding_dim
        # self.c_matrix = Helper.init_from_gaussian(rows=self.v, cols=self.d)
        self.c_matrix = Helper.init_from_uniform(rows=self.v, cols=self.d)
        self.c_grad = np.zeros(shape=self.c_matrix.shape, dtype=np.float32)
        # variables that are updated each epoch
        self.forward_input = None
        self.forward_output = None

    def get_forward_output(self, input_data):
        self.forward_input = input_data
        batch_size = input_data.shape[0]
        feature_size = input_data.shape[1]
        batch_embeddings = np.ndarray((batch_size, feature_size * self.d)).astype(dtype=np.float32)
        for i in range(batch_size):
            single_embeddings = []
            for j in range(feature_size):
                single_embeddings.append(self.c_matrix[int(input_data[i, j]), :])
            batch_embeddings[i] = np.concatenate(single_embeddings)
        self.forward_output = batch_embeddings
        return self.forward_output

    def get_backward_output(self, running_grad):
        self.c_grad = np.zeros(shape=self.c_matrix.shape, dtype=np.float32)
        for i in range(self.forward_input.shape[0]):
            for j in range(self.forward_input.shape[1]):
                word_idx = int(self.forward_input[i, j])
                embd_start_idx = j * self.d
                embd_end_idx = (j + 1) * self.d
                self.c_grad[word_idx, :] += running_grad[i, embd_start_idx: embd_end_idx]

    def update_params(self, lrn_rate):
        self.c_matrix -= lrn_rate * self.c_grad


class FullyConnected(object):
    def __init__(self, rows, cols):
        # self.w = Helper.init_from_gaussian(rows=rows, cols=cols)
        self.w = Helper.init_from_uniform(rows=rows, cols=cols)
        self.b = Helper.init_to_zeros(rows=1, cols=cols)
        # variables that are updated each epoch
        self.forward_input = None
        self.forward_output = None
        self.backward_output = None
        self.w_grad = None
        self.b_grad = None

    def get_forward_output(self, input_data):
        self.forward_input = input_data
        prod = np.dot(input_data, self.w).astype(np.float32)
        self.forward_output = prod + self.b
        return self.forward_output

    def get_backward_output(self, running_grad):
        self.w_grad = np.dot(self.forward_input.T, running_grad).astype(np.float32) / self.forward_input.shape[0]
        self.b_grad = np.mean(running_grad, axis=0, dtype=np.float32)
        self.backward_output = np.dot(running_grad, self.w.T).astype(np.float32)
        return self.backward_output

    def update_params(self, lrn_rate):
        self.w -= lrn_rate * self.w_grad
        self.b -= lrn_rate * self.b_grad


class ActivationType(Enum):
        tanh = 1
        softmax = 2


class Activation(object):
    def __init__(self, activation_type):
        self.activation_type = activation_type
        # variables that are updated each epoch
        self.forward_input = None
        self.forward_output = None
        self.backward_output = None

    def get_forward_output(self, input_data):
        np.clip(input_data, -700, 700)  # Exponent overflow prevention
        self.forward_input = input_data
        if self.activation_type is ActivationType.tanh:
            self.forward_output = np.tanh(self.forward_input).astype(dtype=np.float32)
        elif self.activation_type is ActivationType.softmax:
            h_temp = np.exp(self.forward_input)
            denominators = np.sum(h_temp, axis=1).reshape((np.shape(self.forward_input)[0], 1))
            self.forward_output = h_temp / denominators
        return self.forward_output

    def get_backward_output(self, running_grad):
        if self.activation_type is ActivationType.tanh:
            temp_grad = 1 - np.multiply(self.forward_output, self.forward_output)
            self.backward_output = np.zeros(shape=self.forward_input.shape).astype(np.float32)
            self.backward_output = np.multiply(running_grad, temp_grad)
        elif self.activation_type is ActivationType.softmax:
            self.backward_output = running_grad  # this is gradient of ce loss wrt softmax input
        return self.backward_output


class Loss(object):
    def __init__(self):
        # variables that are updated each epoch
        self.forward_input = None
        self.forward_output = None
        self.backward_output = None

    def get_forward_output(self, input_data, y_batch):
        self.forward_input = input_data
        total_loss = 0
        for i in range(input_data.shape[0]):
            total_loss += np.log(input_data[i, int(y_batch[i])])
        self.forward_output = -1 * total_loss / input_data.shape[0]
        return self.forward_output

    def get_backward_output(self, y_labels, class_count):
        batch_size = self.forward_input.shape[0]
        one_hot_enc = Helper.get_one_hot_encoding(true_labels=y_labels,
                                                  batch_size=batch_size,
                                                  class_count=class_count)
        # gradient of loss wrt prev layer (softmax) input
        self.backward_output = self.forward_input - one_hot_enc
        return self.backward_output


class Helper(object):
    @staticmethod
    def init_from_gaussian(rows, cols):
        mean = 0
        std_dev = 0.1
        return np.random.normal(loc=mean, scale=std_dev, size=(rows, cols))

    @staticmethod
    def init_from_uniform(rows, cols):
        b = math.sqrt(6) / math.sqrt(rows + cols)
        return np.random.uniform(-b, b, (rows, cols)).astype(np.float32)

    @staticmethod
    def init_to_zeros(rows, cols):
        return np.zeros(shape=(rows, cols))

    @staticmethod
    def get_one_hot_encoding(true_labels, batch_size, class_count):
        true_labels = [int(i) for i in true_labels]
        one_hot_enc_matrix = np.zeros((batch_size, class_count), dtype=np.float32)
        one_hot_enc_matrix[np.arange(batch_size), true_labels] = 1
        return one_hot_enc_matrix
