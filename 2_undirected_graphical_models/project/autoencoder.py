import numpy as np
import pandas as pd
from project import data_retriever as dr
import math


class AE(object):
    def __init__(self, h_count, v_count, random_seed, results_file, w=None, h_bias=None, v_bias=None):
        self.h_count = h_count
        self.v_count = v_count
        self.results_file = results_file

        np.random.seed(random_seed)
        normal_mean = 0
        normal_stddev = 0.1

        if w is None:
            self.w = np.random.normal(loc=normal_mean, scale=normal_stddev, size=(self.v_count, self.h_count))
        else:
            self.w = w
        self.w_star = self.w.T

        if h_bias is None:
            self.h_bias = np.zeros((1, h_count))
        else:
            self.h_bias = h_bias

        if v_bias is None:
            self.v_bias = np.zeros((1, v_count))
        else:
            self.v_bias = v_bias

    @staticmethod
    def sigmoid(pre_activation):
        # overflow prevention trick
        pre_activation = np.clip(pre_activation, -700, 700)
        return 1 / (1 + np.exp(-pre_activation))

    def encode(self, visible):
        return AE.sigmoid(np.dot(visible, self.w) + self.h_bias)

    def decode(self, hidden):
        return AE.sigmoid(np.dot(hidden, self.w_star) + self.v_bias)

    @staticmethod
    def get_input_with_dropout(input, dropout):
        return np.random.binomial(n=1, p=1-dropout, size=input.shape) * input

    def perform_forward_prop(self, x, dropout):
        assert 'dropout should be between 0 and 1', 0 <= dropout <= 1
        noisy_data = AE.get_input_with_dropout(input=x, dropout=dropout)
        y = self.encode(visible=noisy_data)
        z = self.decode(hidden=y)
        # using cross entropy reconstruction loss
        epsilon = math.pow(10, -9)
        loss = np.sum(- x * np.log(z + epsilon) - (1 - x) * np.log(1 - z + epsilon), axis=1)
        return y, z, loss

    def perform_back_prop(self, x, y, z):
        dl_by_dz = AE.get_ce_loss_gradient(z=z, x=x)
        dz_by_dpa2 = AE.get_sigmoid_gradient(sigmoid_output=z)
        dl_by_dpa2 = dl_by_dz * dz_by_dpa2
        dl_by_dw = np.dot(y.T, dl_by_dpa2).T
        dl_by_dvbias = dl_by_dpa2

        dl_by_dy = np.dot(dl_by_dpa2, self.w)
        dy_by_dpa1 = AE.get_sigmoid_gradient(sigmoid_output=y)
        dl_by_dpa1 = dl_by_dy * dy_by_dpa1
        dl_by_dw += np.dot(x.T, dl_by_dpa1)
        dl_by_hbias = dl_by_dpa1

        return dl_by_dw, np.mean(dl_by_dvbias, axis=0), np.mean(dl_by_hbias, axis=0)

    @staticmethod
    def get_ce_loss_gradient(z, x):
        return (z-x)/(z*(1-z))

    @staticmethod
    def get_sigmoid_gradient(sigmoid_output):
        return sigmoid_output * (1 - sigmoid_output)

    def update_params(self, w_grad, h_bias_grad, v_bias_grad, lrn_rate):
        self.w -= lrn_rate * w_grad
        self.h_bias -= lrn_rate * h_bias_grad
        self.v_bias -= lrn_rate * v_bias_grad

    def initialize_params_file(self, lrn_rate, mini_batch_size):
        hdf = pd.HDFStore(self.results_file)
        hdf['learning_rate'] = pd.DataFrame([lrn_rate])
        hdf['mini_batch_size'] = pd.DataFrame([mini_batch_size])
        hdf.close()

    def save_params(self, trn_errors=None, vldn_errors=None):
        hdf = pd.HDFStore(self.results_file)
        hdf['w'] = pd.DataFrame(self.w)
        hdf['v_bias'] = pd.DataFrame(self.v_bias)
        hdf['h_bias'] = pd.DataFrame(self.h_bias)
        if trn_errors is not None:
            hdf['trn_ce_recon_err'] = pd.DataFrame(trn_errors)
        if vldn_errors is not None:
            hdf['vldn_ce_recon_err'] = pd.DataFrame(vldn_errors)
        hdf.close()

    def train(self, trn_data, vldn_data, mini_batch_size, lrn_rate, dropout, max_epochs,
              vldn_error_stopping_threshold, vldn_error_checking_window, suppress_output):
        epoch = 0
        iterations_per_epoch = int(trn_data.shape[0] / mini_batch_size)
        all_trn_recon_errors = []
        all_vldn_recon_errors = []
        self.initialize_params_file(lrn_rate=lrn_rate, mini_batch_size=mini_batch_size)

        # stopping criteria variables
        stopping_criteria = True
        prev_avg_vldn_error = float('inf')
        counter = 0

        while stopping_criteria:
            np.random.shuffle(trn_data)
            np.random.shuffle(vldn_data)

            for i in range(1, iterations_per_epoch):
                start_idx = (i - 1) * mini_batch_size
                end_idx = i * mini_batch_size
                mini_batch = trn_data[start_idx:end_idx, :]

                # forward pass
                y, z, _ = self.perform_forward_prop(x=mini_batch, dropout=dropout)

                # back propagation and weight update
                [w_grad, v_bias_grad, h_bias_grad] = self.perform_back_prop(x=mini_batch, y=y, z=z)
                self.update_params(w_grad, h_bias_grad, v_bias_grad, lrn_rate)

            epoch += 1

            # calculate errors
            _, _, trn_loss = self.perform_forward_prop(x=trn_data, dropout=0)
            _, _, vldn_loss = self.perform_forward_prop(x=vldn_data, dropout=0)
            avg_trn_error = np.mean(trn_loss)
            avg_vldn_error = np.mean(vldn_loss)
            all_trn_recon_errors.append(np.mean(trn_loss))
            all_vldn_recon_errors.append(avg_vldn_error)

            # stopping criteria
            if epoch == max_epochs:
                stopping_criteria = False

            if prev_avg_vldn_error - avg_vldn_error <= vldn_error_stopping_threshold:
                counter += 1
                if counter == vldn_error_checking_window:
                    stopping_criteria = False
            else:
                counter = 0

            prev_avg_vldn_error = avg_vldn_error

            if not suppress_output:
                print("Epoch %s, avg trn ce recon error = %s, avg vldn cs recon error = %s" % (epoch, avg_trn_error, avg_vldn_error))

            # finally save params
            self.save_params()

        self.save_params(trn_errors=all_trn_recon_errors, vldn_errors=all_vldn_recon_errors)


if __name__ == '__main__':
    trn_data, _, vldn_data, _, test_data, _ = dr.DataRetriever.get_data(data_folder='data')
    ae = AE(h_count=100, v_count=trn_data.shape[1], random_seed=2017, results_file='results/ae/resultx.hdf5')
    ae.train(trn_data=trn_data,
             vldn_data=vldn_data,
             mini_batch_size=32,
             lrn_rate=0.0001,
             dropout=0.5,
             max_epochs=1000,
             vldn_error_stopping_threshold=0.1,
             vldn_error_checking_window=10,
             suppress_output=False)
