import numpy as np
import pandas as pd
import copy
from project import data_retriever as dr
import time


class RBM(object):

    def __init__(self, h_count, v_count, random_seed, results_file, w=None, h_bias=None, v_bias=None):
        self.h_count = h_count
        self.v_count = v_count
        self.results_file = results_file

        np.random.seed(random_seed)
        normal_mean = 0
        normal_stddev = 0.1

        if w is None:
            self.w = np.random.normal(loc=normal_mean,
                                      scale=normal_stddev,
                                      size=(self.v_count, self.h_count))
        else:
            self.w = w

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
        return 1/(1+np.exp(-pre_activation))

    def get_hidden_probs(self, v_units):
        pre_activation = np.dot(v_units, self.w) + self.h_bias
        return RBM.sigmoid(pre_activation)

    def get_visible_probs(self, h_units):
        pre_activation = np.dot(h_units, self.w.T) + self.v_bias
        return RBM.sigmoid(pre_activation)

    @staticmethod
    def sample_one_given_other(prob):
        units = np.zeros(shape=prob.shape, dtype=int)
        random_prob_samples = np.random.rand(prob.shape[0], prob.shape[1])
        flip_locations = random_prob_samples < prob
        units[flip_locations] = 1
        return units

    def perform_cd(self, v_units, cd_k):
        assert "The minimum value of cd_k is 1", cd_k > 0
        neg_v_units = v_units

        for i in range(cd_k):
            h_prob = self.get_hidden_probs(v_units=neg_v_units)
            h_units = RBM.sample_one_given_other(prob=h_prob)
            # Store the sigmoid activations corresponding to the first visible units (i.e. training data)
            if i == 0:
                initial_v_units_activation = h_prob
            v_prob = self.get_visible_probs(h_units=h_units)
            if i == cd_k-1:
                reconstruction_prob = v_prob
            neg_v_units = RBM.sample_one_given_other(prob=v_prob)

        neg_v_activation = self.get_hidden_probs(v_units=neg_v_units)

        return initial_v_units_activation, neg_v_units, neg_v_activation, reconstruction_prob

    def update_params(self, av_trn_visible, av_trn_sigmoid, av_neg_visible, av_neg_sigmoid, lrn_rate):
        self.w += lrn_rate*(np.dot(av_trn_visible.T, av_trn_sigmoid) - (np.dot(av_neg_visible.T, av_neg_sigmoid)))
        self.v_bias += lrn_rate*(av_trn_visible - av_neg_visible)
        self.h_bias += lrn_rate*(av_trn_sigmoid - av_neg_sigmoid)

    def vhv_gibbs_sample(self, v0_sample):
        h_prob = self.get_hidden_probs(v_units=v0_sample)
        h_units = RBM.sample_one_given_other(prob=h_prob)
        v_prob = self.get_visible_probs(h_units=h_units)
        return RBM.sample_one_given_other(prob=v_prob)

    def train(self, trn_data, vldn_data, mini_batch_size,
              cd_k, lrn_rate, max_epochs, vldn_error_stopping_threshold,
              vldn_error_checking_window, suppress_output):

        start_time = time.time()

        epoch = 0
        iterations_per_epoch = int(trn_data.shape[0] / mini_batch_size)
        all_trn_recon_errors = []
        all_vldn_recon_errors = []
        self.initialize_params_file(cd_k=cd_k, lrn_rate=lrn_rate, mini_batch_size=mini_batch_size)

        # Free-energy calculation
        np.random.shuffle(trn_data)
        ovftng_criteria_trn_data = copy.deepcopy(trn_data[:100, :])
        curr_fe_diff = self.get_abs_avg_free_energy_diff(dataset_1=ovftng_criteria_trn_data, dataset_2=vldn_data)
        fe_diff = [curr_fe_diff]

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
                trn_sigmoid, neg_v_units, neg_v_sigmoid, _ = self.perform_cd(v_units=mini_batch, cd_k=cd_k)
                self.update_params(av_trn_visible=np.asmatrix(np.mean(mini_batch, axis=0)),
                                   av_trn_sigmoid=np.asmatrix(np.mean(trn_sigmoid, axis=0)),
                                   av_neg_visible=np.asmatrix(np.mean(neg_v_units, axis=0)),
                                   av_neg_sigmoid=np.asmatrix(np.mean(neg_v_sigmoid, axis=0)),
                                   lrn_rate=lrn_rate)

            epoch += 1

            # calculate errors
            [avg_trn_error, avg_vldn_error] = self.calc_reconstruction_error(trn_data=trn_data, vldn_data=vldn_data, cd_k=cd_k)
            all_trn_recon_errors.append(avg_trn_error)
            all_vldn_recon_errors.append(avg_vldn_error)

            # free energy update
            curr_fe_diff = self.get_abs_avg_free_energy_diff(dataset_1=ovftng_criteria_trn_data, dataset_2=vldn_data)
            fe_diff.append(curr_fe_diff)

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
                print("Epoch %s, avg trn ce recon error = %s, avg vldn cs recon error = %s, fe_diff = %s"
                      % (epoch, avg_trn_error, avg_vldn_error, curr_fe_diff))

            self.save_params()

        self.save_params(trn_errors=all_trn_recon_errors, vldn_errors=all_vldn_recon_errors, free_energy_diffs=fe_diff)
        print("Total time taken to train = %s" % (time.time() - start_time))

    def get_free_energy(self, v_sample):
        exp_term = np.dot(v_sample, self.w) + self.h_bias
        not_log_term = np.dot(v_sample, self.v_bias.T)
        log_term = np.sum(np.log(1 + np.exp(exp_term)), axis=1)
        return - not_log_term - log_term

    def get_abs_avg_free_energy_diff(self, dataset_1, dataset_2):
        fe1 = np.mean(self.get_free_energy(v_sample=dataset_1))
        fe2 = np.mean(self.get_free_energy(v_sample=dataset_2))
        return np.absolute(fe1 - fe2)

    def calc_reconstruction_error(self, trn_data, vldn_data, cd_k):
        _, _, _, trn_reconstruction_prob = self.perform_cd(v_units=trn_data, cd_k=cd_k)
        _, _, _, vldn_reconstruction_prob = self.perform_cd(v_units=vldn_data, cd_k=cd_k)

        assert trn_data.shape == trn_reconstruction_prob.shape
        assert vldn_data.shape == vldn_reconstruction_prob.shape

        trn_temp_error = - trn_data * np.log(trn_reconstruction_prob) - (1-trn_data) * np.log(1-trn_reconstruction_prob)
        vldn_temp_error = - vldn_data * np.log(vldn_reconstruction_prob) - (1-vldn_data) * np.log(1 - vldn_reconstruction_prob)
        trn_error = np.mean(np.sum(trn_temp_error, axis=1))
        vldn_error = np.mean(np.sum(vldn_temp_error, axis=1))

        return trn_error, vldn_error

    def initialize_params_file(self, cd_k, lrn_rate, mini_batch_size):
        hdf = pd.HDFStore(self.results_file)
        hdf['cd_k'] = pd.DataFrame([cd_k])
        hdf['learning_rate'] = pd.DataFrame([lrn_rate])
        hdf['mini_batch_size'] = pd.DataFrame([mini_batch_size])
        hdf.close()

    def save_params(self, trn_errors=None, vldn_errors=None, free_energy_diffs=None):
        hdf = pd.HDFStore(self.results_file)
        hdf['w'] = pd.DataFrame(self.w)
        hdf['v_bias'] = pd.DataFrame(self.v_bias)
        hdf['h_bias'] = pd.DataFrame(self.h_bias)
        if trn_errors is not None:
            hdf['trn_ce_recon_err'] = pd.DataFrame(trn_errors)
        if vldn_errors is not None:
            hdf['vldn_ce_recon_err'] = pd.DataFrame(vldn_errors)
        if free_energy_diffs is not None:
            hdf['free_energies_diff'] = pd.DataFrame(free_energy_diffs)
        hdf.close()


if __name__ == '__main__':
    trn_data, _, vldn_data, _, test_data, _ = dr.DataRetriever.get_data(data_folder='data')
    #  train the model
    rbm = RBM(h_count=100, v_count=trn_data.shape[1], random_seed=2017, results_file='results/rbm/rbm_deg.hdf5')
    rbm.train(trn_data=trn_data,
              vldn_data=vldn_data,
              mini_batch_size=32,
              cd_k=20,
              lrn_rate=0.1,
              max_epochs=1000,
              vldn_error_stopping_threshold=0.1,
              vldn_error_checking_window=10,
              suppress_output=False)
