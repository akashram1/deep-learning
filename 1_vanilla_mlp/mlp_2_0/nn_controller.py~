import copy
import pickle
import os.path as ospath
import data_retriever as dr
import nn_core_logic as ncl


def get_nn_struc(hidden_layers_info):

    layers = {}
    total_hidden_layers = len(hidden_layers_info)

    # Filling up hidden layers
    for i in range(1, total_hidden_layers + 1):
        layers[i] = {}
        layers[i]['TYPE'] = "HIDDEN"
        layers[i]['UNITS'] = hidden_layers_info[i]["UNITS"]
        layers[i]['ACTIVATION_FUNCTION'] = hidden_layers_info[i]["ACTIVATION_FUNCTION"]

    # Output layer
    layers[total_hidden_layers + 1] = {}
    layers[total_hidden_layers + 1]['TYPE'] = "OUTPUT"
    layers[total_hidden_layers + 1]['UNITS'] = 10  # Since this is output layer, units is inferred as number of classes
    # It's inferred that output layer uses softmax function

    # Loss layer
    layers[total_hidden_layers + 2] = {}
    layers[total_hidden_layers + 2]['TYPE'] = "LOSS"
    layers[total_hidden_layers + 2]['LOSS_TYPE'] = "CE"

    return layers


def run_nn(batch_size, max_epochs, learning_rate, momentum, reg_decay, early_stopping_patience_limit, hidden_layers_info, seed, pickle_file_destination, w, suppress_output):
    # Get the data
    x_train, y_train, x_valid, y_valid, x_test, y_test = dr.DataRetriever.get_data('project/data')
    y_train = np.reshape(y_train, newshape=(x_train.shape[0], 1))
    y_valid = np.reshape(y_valid, newshape=(x_valid.shape[0], 1))
    y_test = np.reshape(y_test, newshape=(x_test.shape[0], 1))

    train_combined = np.concatenate((x_train, y_train), axis=1)
    valid_combined = np.concatenate((x_valid, y_valid), axis=1)
    test_combined = np.concatenate((x_test, y_test), axis=1)

    # Set the hyper-parameters
    training_size = np.shape(train_combined)[0]
    class_count = len(np.unique(train_combined[:, -1]))
    features_per_image = np.shape(train_combined)[1] - 1

    iterations_per_epoch = int(training_size / batch_size)

    np.random.seed(seed)

    # Obtain the structure of the nn
    layers = get_nn_struc(hidden_layers_info)

    # Initialize the parameters using random seed
    params = ncl.init_nn(layers=layers, features_per_image=features_per_image, w=w)

    # Create a dictionary to store this specific experiment
    experiment_summary = {'lr': learning_rate,
                          'momentum': momentum,
                          'l2_reg_decay': reg_decay,
                          'seed': seed,
                          'batch_size': batch_size,
                          'epochs': [],
                          'avg_trn_loss': [],
                          'avg_vdn_loss': [],
                          'training_acc': [],
                          'valid_acc': [],
                          'params': copy.deepcopy(params)}

    # Early stopping parameters
    global j
    global prev_val_loss
    prev_val_loss = float('inf')
    j = 0

    # Start the learning
    for epoch in range(1, max_epochs + 1):

        np.random.shuffle(train_combined)
        np.random.shuffle(valid_combined)
        np.random.shuffle(test_combined)

        x_valid = valid_combined[:, :-1]
        y_valid = valid_combined[:, -1]

        for i in range(1, iterations_per_epoch + 1):
            start_idx = (i - 1) * batch_size
            end_idx = i * batch_size

            x_minibatch = train_combined[start_idx:end_idx, :-1]
            y_minibatch = train_combined[start_idx:end_idx, -1]
            one_hot_true_labels_matrix = ncl.get_one_hot_encoding(true_labels=y_minibatch, batch_size=batch_size, class_count=class_count)

            # forward pass
            training_output_tensor = ncl.forward_pass_controller(x=x_minibatch, one_hot_true_labels_matrix=one_hot_true_labels_matrix, layers=layers, params=params)
            # backward pass
            params_grad = ncl.backward_pass_controller(output_tensor=training_output_tensor, one_hot_true_labels_matrix=one_hot_true_labels_matrix, layers=layers, params=params)
            # update params
            params = ncl.update_params(learning_rate=learning_rate, decay=reg_decay, momentum=momentum,  params=params, params_grad=params_grad)

        # training accuracy and loss on the whole training set
        x_whole_train = train_combined[:, :-1]
        y_whole_train = train_combined[:, -1]
        one_hot_true_labels_matrix = ncl.get_one_hot_encoding(true_labels=y_whole_train, batch_size=len(y_whole_train), class_count=class_count)
        training_output_tensor = ncl.forward_pass_controller(x=x_whole_train, one_hot_true_labels_matrix=one_hot_true_labels_matrix, layers=layers, params=params)

        avg_training_loss = training_output_tensor[len(layers)]
        training_acc = ncl.get_classification_acc(true_labels=y_whole_train, prob=training_output_tensor[len(layers) - 1])

        # validation accuracy on the whole validation set
        one_hot_true_labels_matrix = ncl.get_one_hot_encoding(true_labels=y_valid, batch_size=len(y_valid), class_count=class_count)
        validation_output_tensor = ncl.forward_pass_controller(x=x_valid, one_hot_true_labels_matrix=one_hot_true_labels_matrix, layers=layers, params=params)
        avg_validation_loss = validation_output_tensor[len(layers)]
        validation_acc = ncl.get_classification_acc(true_labels=y_valid, prob=validation_output_tensor[len(layers) - 1])

        # pickle all params to prevent loss
        experiment_summary['epochs'].append(epoch)
        experiment_summary['training_acc'].append(training_acc)
        experiment_summary['valid_acc'].append(validation_acc)

        experiment_summary['avg_trn_loss'].append(avg_training_loss)
        experiment_summary['avg_vdn_loss'].append(avg_validation_loss)

        experiment_summary['params'] = copy.deepcopy(params)
        experiment_summary['hidden_layers_summary'] = hidden_layers_info

        with open(pickle_file_destination, 'wb') as handle:
            pickle.dump(experiment_summary, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if not suppress_output:
            print("epoch = %s, avg_training_loss = %s, avg_vdn_loss = %s,  training_acc = %s, validation_acc = %s"
                  % (epoch, avg_training_loss, avg_validation_loss,  training_acc,  validation_acc))

        if avg_validation_loss < prev_val_loss:
            j = 0
            prev_val_loss = avg_validation_loss
        else:
            j += 1
            if j == early_stopping_patience_limit:
                break
    # test accuracy

    x_test = test_combined[:, :-1]
    y_test = test_combined[:, -1]
    one_hot_true_labels_matrix = ncl.get_one_hot_encoding(true_labels=y_test, batch_size=len(y_test), class_count=class_count)
    test_output_tensor = ncl.forward_pass_controller(x=x_test, one_hot_true_labels_matrix=one_hot_true_labels_matrix, layers=layers, params=params)
    avg_test_loss = test_output_tensor[len(layers)]
    test_acc = ncl.get_classification_acc(true_labels=y_test, prob=test_output_tensor[len(layers) - 1])
    experiment_summary['test_acc'] = test_acc
    experiment_summary['avg_test_loss'] = avg_test_loss

    with open(pickle_file_destination, 'wb') as handle:
        pickle.dump(experiment_summary, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return experiment_summary


if __name__ == '__main__':
    # CHANGE THESE WITHOUT FAIL FOR EVERY EXPERIMENT:
    ##################################################
    expt_id = "results/relu.pickle"
    ###################################################
    hidden_layers_dict = {1: {"UNITS": 100, "ACTIVATION_FUNCTION": "SIGMOID"}}

    # set w
    w = np.ones((784, 100))
    experiment = run_nn(batch_size=32,
                        max_epochs=200,
                        learning_rate=0.5,
                        hidden_layers_info=hidden_layers_dict,
                        momentum=0,
                        reg_decay=0,
                        early_stopping_patience_limit=15,
                        seed=2017,
                        pickle_file_destination=expt_id,
                        w=w,
                        suppress_output=True)
