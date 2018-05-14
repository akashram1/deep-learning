## Restricted Boltzmann Machine

1. Go the main function of rbm.py
2. Change the data_folder argument to the path to the 3 data input files without a tailing '/'.

```
trn_data, _, vldn_data, _, test_data, _ = dr.DataRetriever.get_data(data_folder='data')
```

3.Initialize the RBM
rbm1 = rbm.RBM(h_count=100, v_count=x_train.shape[1], random_seed=2017, results_file=rbm_results_path + 'rbm1.hdf5')

h_count is the number of hidden units
v_count is number of visible units
results_file is the path to the hdf5 file in which the results of this process are stored.

4. Train the RBM
    rbm.train(trn_data=trn_data,
              vldn_data=vldn_data,
              mini_batch_size=32,
              cd_k=20,
              lrn_rate=0.1,
              max_epochs=1000,
              vldn_error_stopping_threshold=0.1,
              vldn_error_checking_window=10,
              suppress_output=False)

The parameters are self explanatory. Set suppress_ouput to True for the losses to be displayed every epoch.
               
## Autoencoder and Denoising Autoencoder
1. Go to the main function of autoencoder.py
2. Same as RBM
3. Initialize the AE
ae = AE(h_count=100, v_count=trn_data.shape[1], random_seed=2017, results_file='results/ae/result1.hdf5')
4. Train the AE: The parameters are self explanatory
    ae.train(trn_data=trn_data,
             vldn_data=vldn_data,
             mini_batch_size=32,
             lrn_rate=0.01,
             dropout=0.0,
             max_epochs=1000,
             vldn_error_stopping_threshold=0.1,
             vldn_error_checking_window=10,
             suppress_output=False)

Note that an Autoencoder becomes a Denoising Autoencoder for > 0 values of dropout. Dropout is a fraction between 0 and 1.
