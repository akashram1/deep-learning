import numpy as np
import nn_core_logic as ncl


def get_nn_struc():

    layers = {1: {}, 2: {}, 3: {}}

    # Hidden layer
    layers[1]['TYPE'] = "HIDDEN"
    layers[1]['UNITS'] = 100
    layers[1]['ACTIVATION_FUNCTION'] = "SIGMOID"

    # Output layer
    layers[2]['TYPE'] = "OUTPUT"
    layers[2]['UNITS'] = 10 # Since this is output layer, units is inferred as number of classes
    # It's inferred that output layer uses softmax function

    # Loss layer
    layers[3]['TYPE'] = "LOSS"
    layers[3]['LOSS_TYPE'] = "CE"

    return layers


def main():
    # Get the data
    train_combined = np.loadtxt('digitstrain.txt', delimiter=',')
    valid_combined = np.loadtxt('digitsvalid.txt', delimiter=',')
    x_test = np.loadtxt('digitstest.txt', delimiter=',')

    # Set the hyper-parameters
    training_size = np.shape(train_combined)[0]
    batch_size = 64
    class_count = 10
    features_per_image = np.shape(train_combined)[1] - 1

    min_epochs = 200
    max_epochs = 5000
    iterations_per_epoch = int(training_size / batch_size)

    test_interval = 500
    display_interval = 100
    snapshot = 5000

    learning_rate = 0.1
    momentum = 0.5

    seed = 2
    np.random.seed(seed)

    # Obtain the structure of the nn
    layers = get_nn_struc()

    # Initialize the parameters using random seed
    params = ncl.init_nn(layers=layers, features_per_image=features_per_image, sigma=0.01, mean=0)

    # Start the learning
    for epoch in range(1, max_epochs):

        np.random.shuffle(train_combined)
        np.random.shuffle(valid_combined)
        np.random.shuffle(x_test)

        x_valid = valid_combined[:, :-1]
        y_valid = valid_combined[:, -1]

        for i in range(1, iterations_per_epoch+1):
            start_idx = (i - 1) * batch_size
            end_idx = i * batch_size

            x_train = train_combined[start_idx:end_idx, :-1]
            y_train = train_combined[start_idx:end_idx, -1]

            [total_loss, train_acc, params_grad] = ncl.single_forward_backward_pass(x_train=x_train,
                                                                                    y_train=y_train,
                                                                                    layers=layers,
                                                                                    params=params,
                                                                                    class_count= class_count)

            params = ncl.update_params(learning_rate=learning_rate, params=params, params_grad=params_grad)

            if ((epoch-1)*iterations_per_epoch + i) % display_interval == 0:
                print(epoch, (epoch-1)*iterations_per_epoch + i, total_loss, train_acc)

if __name__ == '__main__':
    main()
