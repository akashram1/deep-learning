import numpy as np
import pandas as pd
from project import mlp_layers as layer_types
import time
import pickle

class LanguageMLP(object):
    def __init__(self):
        self.layers = []
        self.vldn_perplexity = []
        self.trn_ce = []
        self.vldn_ce = []
        self.train_params = {}

    def add_embedding(self, vocab_size, embedding_dim):
        self.layers.append(layer_types.Embedding(vocab_size=vocab_size, embedding_dim=embedding_dim))

    def add_fully_connected(self, input_cols, dim):
        self.layers.append(layer_types.FullyConnected(rows=input_cols, cols=dim))

    def add_activation(self, activn):
        assert "Incorrect activation type", type(activn) == layer_types.ActivationType
        self.layers.append(layer_types.Activation(activn))

    def add_loss(self):
        self.layers.append(layer_types.Loss())

    def perform_forward_pass(self, x_batch, y_batch):
        input_data = x_batch
        for layer in self.layers:
            if type(layer) is not layer_types.Loss:
                output_data = layer.get_forward_output(input_data=input_data)
            else:
                output_data = layer.get_forward_output(input_data, y_batch)
            input_data = output_data
        return input_data  # avg cross entropy

    def perform_backward_pass(self, y_labels, class_count):
        running_grad = None
        for layer in reversed(self.layers):
            if type(layer) is not layer_types.Loss:
                running_grad = layer.get_backward_output(running_grad=running_grad)
            else:
                running_grad = layer.get_backward_output(y_labels=y_labels, class_count=class_count)

    def update_params(self, lrn_rate):
        for layer in self.layers:
            if type(layer) is layer_types.FullyConnected or type(layer) is layer_types.Embedding:
                layer.update_params(lrn_rate=lrn_rate)

    def train(self, trn_data, vldn_data, results_file,
              batch_size, class_count, lrn_rate,
              max_epochs, suppress_output):
        self.train_params['lrn_rate'] = lrn_rate
        self.train_params['batch_size'] = batch_size
        self.train_params['results_file'] = results_file

        start_time = time.time()
        epoch = 0
        iterations_per_epoch = int(trn_data.shape[0] / batch_size)

        # stopping criteria variables
        stopping_criteria = True

        while stopping_criteria:
            np.random.shuffle(trn_data)
            np.random.shuffle(vldn_data)

            for i in range(1, iterations_per_epoch + 1):
                start_idx = (i - 1) * batch_size
                end_idx = i * batch_size
                x_batch = trn_data[start_idx:end_idx, :- 1]
                y_batch = trn_data[start_idx:end_idx, -1]
                loss = self.perform_forward_pass(x_batch, y_batch)
                self.perform_backward_pass(y_labels=y_batch, class_count=class_count)
                self.update_params(lrn_rate=lrn_rate)

            epoch += 1

            # calculate training loss - can't pass whole trn set at once due to memory errors
            total_avg_trn_ce = 0
            for i in range(1, iterations_per_epoch + 1):
                start_idx = (i - 1) * batch_size
                end_idx = i * batch_size
                x_batch = trn_data[start_idx:end_idx, :- 1]
                y_batch = trn_data[start_idx:end_idx, -1]
                loss = self.perform_forward_pass(x_batch, y_batch)
                total_avg_trn_ce += loss

            avg_trn_ce = total_avg_trn_ce / float(iterations_per_epoch)
            avg_vldn_ce = self.perform_forward_pass(x_batch=vldn_combined[:, :-1], y_batch=vldn_combined[:, -1])
            vldn_perplexity = np.exp(avg_vldn_ce)
            self.vldn_perplexity.append(vldn_perplexity)
            self.trn_ce.append(avg_trn_ce)
            self.vldn_ce.append(avg_vldn_ce)

            # stopping criteria
            if epoch == max_epochs:
                stopping_criteria = False

            if not suppress_output:
                print("epoch %s, time (min) = %s, avg trn ce = %s, avg vldn ce = %s, vldn perplexity = %s"
                      % (epoch, (time.time()-start_time)/60.0, avg_trn_ce, avg_vldn_ce, vldn_perplexity))

            pickle.dump(self, open(self.train_params['results_file'], 'wb'))

        print("Total time taken to train in hours = %s" % ((time.time() - start_time)/3600.0))


if __name__ == '__main__':
    # paths
    data_path = 'data/'
    results_path = 'results/'

    # pre-reqs
    np.random.seed(2017)
    trn_combined = np.asarray(pd.read_hdf(data_path + 'trn_combined.hdf5'))
    vldn_combined = np.asarray(pd.read_hdf(data_path + 'vldn_combined.hdf5'))
    vocab_to_id = dict(pd.read_hdf(data_path + 'vocab_to_id.hdf5'))
    vocab_size = len(vocab_to_id)
    embedding_size = 16
    # change these ONLY!
    h1 = 256
    rf = '3_2_256_f.pickle'
    #####
    h2 = vocab_size
    n_gram_n = trn_combined.shape[1]

    nn = LanguageMLP()
    nn.add_embedding(vocab_size=vocab_size, embedding_dim=embedding_size)
    nn.add_fully_connected(input_cols=embedding_size*(n_gram_n-1), dim=h1)
    # nn.add_activation(activn=layer_types.ActivationType.tanh)
    nn.add_fully_connected(input_cols=h1, dim=h2)
    nn.add_activation(activn=layer_types.ActivationType.softmax)
    nn.add_loss()  # implicitly cross entropy

    nn.train(trn_data=trn_combined,
             vldn_data=vldn_combined,
             results_file=results_path + rf,
             batch_size=512,
             class_count=vocab_size,
             lrn_rate=0.01,
             max_epochs=100,
             suppress_output=False)
