import numpy as np
import math
import copy


def init_nn(layers, features_per_image, mean, sigma):

    params = {}
    for i in range(1, len(layers)):   # All layers but the input (layer 0) and final loss layer have weights and biases
        params[i] = {}
        if i == 1:
            input_cols = features_per_image
        else:
            input_cols = layers[i-1]["UNITS"]

        num_hidden_units = layers[i]["UNITS"]
        params[i]['W'] = np.random.normal(mean, sigma, (input_cols, num_hidden_units))
        params[i]['b'] = np.zeros(shape=(1, num_hidden_units))

    return params


def single_forward_backward_pass(x_train, y_train, layers, params, class_count):

    # Forward Pass
    output_tensor = {0: x_train}
    y = get_one_hot_encoding(true_labels=y_train, batch_size=np.shape(x_train)[0], class_count=class_count)
    for i in range(1, len(layers) + 1):
        if layers[i]["TYPE"] == "HIDDEN":
            output_tensor[i] = hidden_forward_pass(input_tensor=output_tensor[i-1],
                                                   params=params[i],
                                                   activation_function=layers[i]['ACTIVATION_FUNCTION'])

        elif layers[i]["TYPE"] == "OUTPUT":
            output_tensor[i] = output_forward_pass(input_tensor=output_tensor[i - 1],
                                                   params=params[i])

        elif layers[i]["TYPE"] == "LOSS":
            output_tensor[i] = loss_forward_pass(input_tensor=output_tensor[i - 1],
                                                 one_hot_encoded_true_labels=y,
                                                 loss_type=layers[i]['LOSS_TYPE'])

    average_loss = output_tensor[len(layers)]
    training_acc = get_training_acc(true_labels=y_train, prob=output_tensor[len(layers)-1])

    # Backward Pass
    params_grad = {}
    for i in range(len(layers), 0, -1):
        if layers[i]['TYPE'] == "LOSS":
            params_grad[i] = loss_backward_pass(input_tensor=output_tensor[i-1],
                                                one_hot_encoded_true_labels=y,
                                                loss_type=layers[i]['LOSS_TYPE'])

        elif layers[i]['TYPE'] == "OUTPUT":
            params_grad[i] = output_backward_pass(output_grad=params_grad[i+1]['INPUT_GRAD'],
                                                  params=params[i],
                                                  input_tensor=output_tensor[i-1])

        elif layers[i]['TYPE'] == "HIDDEN":
            params_grad[i] = hidden_backward_pass(output_grad=params_grad[i + 1]['INPUT_GRAD'],
                                                  output_tensor=output_tensor[i],
                                                  params=params[i],
                                                  activation_function=layers[i]['ACTIVATION_FUNCTION'],
                                                  input_tensor=output_tensor[i - 1])

    return average_loss, training_acc, params_grad


def hidden_forward_pass(input_tensor, params, activation_function):
    W = params['W']
    b = params['b']
    assert np.shape(input_tensor)[1] == np.shape(W)[0], "Dimension mismatch between X and W in hidden layer"

    a = np.dot(input_tensor, W) + b

    if activation_function == "SIGMOID":
        # Overflow prevention trick
        # exp_overflow_limit = 500
        # if np.max(a) > exp_overflow_limit:
        #     a = np.clip(a, -exp_overflow_limit, exp_overflow_limit)

        output = 1/(1 + np.exp(a))

    return output


def output_forward_pass(input_tensor, params):
    W = params['W']
    b = params['b']

    assert np.shape(input_tensor)[1] == np.shape(W)[0], "Dimension mismatch between X and W in output layer"

    a = np.dot(input_tensor, W) + b

    # Overflow prevention trick
    for img in a:
        img = img - np.max(img)

    # Softmax calculation
    h_temp = np.exp(a)
    denominators = np.sum(h_temp, axis=1).reshape((np.shape(a)[0], 1))
    y_hat = h_temp / denominators

    assert np.shape(y_hat) == (np.shape(input_tensor)[0], np.shape(W)[1]), "Wrong dimension of output of output layer"

    return y_hat


def loss_forward_pass(input_tensor, one_hot_encoded_true_labels, loss_type):

    y_hat = input_tensor
    assert np.shape(y_hat) == np.shape(one_hot_encoded_true_labels), \
        "softmax output and one-hot encoded true label matrix not of same shape!"

    if loss_type == "CE":
        temp = y_hat + math.pow(10, -10)
        log_y_hat = -1 * np.log(temp)
        loss_matrix = np.multiply(log_y_hat, one_hot_encoded_true_labels)
        avg_loss = np.sum(loss_matrix)/np.shape(input_tensor)[0]

    return avg_loss


def loss_backward_pass(input_tensor, one_hot_encoded_true_labels, loss_type):
    if loss_type == "CE":
        assert np.shape(input_tensor) == np.shape(one_hot_encoded_true_labels), \
            "softmax output and one-hot encoded matrix not of same shape!"
        temp = np.sum(input_tensor - one_hot_encoded_true_labels, axis=0)
        grad = (temp / np.shape(one_hot_encoded_true_labels)[0]).reshape((1, np.shape(one_hot_encoded_true_labels)[1]))
        return {'INPUT_GRAD': grad}


def output_backward_pass(output_grad, params, input_tensor):

    W = params['W']
    b = params['b']

    averaged_input = np.mean(input_tensor, axis=0).reshape((1, np.shape(input_tensor)[1]))

    param_grad = {'W': np.outer(averaged_input, output_grad), 'b': output_grad, 'INPUT_GRAD': np.dot(output_grad, W.T)}

    assert np.shape(param_grad['W']) == np.shape(W), "weight gradient and weight matrix shapes mismatch in output layer"
    assert np.shape(param_grad['b']) == np.shape(b), "bias gradient and bias matrix shapes mismatch in output layer"
    assert np.shape(param_grad['INPUT_GRAD']) == (1, np.shape(input_tensor)[1]), "input gradient and input matrix shape " \
                                                                          "mismatch in output layer"

    return param_grad


def hidden_backward_pass(output_grad, output_tensor, params, activation_function, input_tensor):
    W = params['W']
    b = params['b']

    if activation_function == "SIGMOID":

        averaged_activated_output = np.mean(output_tensor, axis=0).reshape((1, np.shape(output_tensor)[1]))

        temp = np.multiply(averaged_activated_output, 1 - averaged_activated_output)
        assert np.shape(temp) == (1, np.shape(output_tensor)[1])
        updated_output_grad = np.multiply(output_grad, temp)

        averaged_input = np.mean(input_tensor, axis=0).reshape((1, np.shape(input_tensor)[1]))

        param_grad = {'W': np.outer(averaged_input,  updated_output_grad), 'b': updated_output_grad,
                      'INPUT': np.dot(updated_output_grad, W.T)}

    assert np.shape(param_grad['W']) == np.shape(W), "weight gradient and weight matrix shapes mismatch in output layer"
    assert np.shape(param_grad['b']) == np.shape(b), "bias gradient and bias matrix shapes mismatch in output layer"
    assert np.shape(param_grad['INPUT']) == (1, np.shape(input_tensor)[1]), \
        "input gradient and input matrix shapes mismatch in output layer"

    return param_grad


def get_one_hot_encoding(true_labels, batch_size, class_count):

    true_labels = [int(i) for i in true_labels]
    one_hot_matrix = np.zeros((batch_size, class_count), dtype=int)
    one_hot_matrix[np.arange(batch_size), true_labels] = 1

    return one_hot_matrix


def get_training_acc(true_labels, prob):
    x = []
    for i in range(np.shape(prob)[0]):
        if np.argmax(prob[i]) == true_labels[i]:
            x.append(1)

    return len(x)/np.shape(prob)[0]


def update_params(learning_rate, params, params_grad):
    params_new = copy.deepcopy(params)
    for i in range(1, len(params)+1):
        params_new[i]['W'] = params[i]['W'] - (learning_rate * params_grad[i]['W'])
        params_new[i]['b'] = params[i]['b'] - (learning_rate * params_grad[i]['b'])

    return params_new
