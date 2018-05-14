import numpy as np
import math
import copy


def init_nn(layers, features_per_image, w):

    params = {}

    for i in range(1, len(layers)):   # All layers but the input (layer 0) and final loss layer have weights and biases
        params[i] = {}
        if i == 1:
            input_cols = features_per_image
        else:
            input_cols = layers[i-1]["UNITS"]

        num_hidden_units = layers[i]["UNITS"]

        # Initialization technique talked about in class
        if i == 1:
            params[i]['W'] = w
            params[i]['b'] = np.zeros(shape=(1, num_hidden_units))
        else:
            b = math.sqrt(6)/math.sqrt(num_hidden_units + input_cols)
            params[i]['W'] = np.random.uniform(-b, b, (input_cols, num_hidden_units))
            params[i]['b'] = np.zeros(shape=(1, num_hidden_units))

        # Initialize the velocity matrices
        params[i]['V_W'] = np.zeros(shape=(input_cols, num_hidden_units))
        params[i]['V_b'] = np.zeros(shape=(1, num_hidden_units))

    return params


def forward_pass_controller(x, one_hot_true_labels_matrix, layers, params):
    output_tensor = {0: x}

    for i in range(1, len(layers) + 1):
        if layers[i]["TYPE"] == "HIDDEN":
            output_tensor[i] = hidden_forward_pass(input_tensor=output_tensor[i - 1], params=params[i], activation_function=layers[i]['ACTIVATION_FUNCTION'])

        elif layers[i]["TYPE"] == "OUTPUT":
            output_tensor[i] = output_forward_pass(input_tensor=output_tensor[i - 1], params=params[i])

        elif layers[i]["TYPE"] == "LOSS":
            output_tensor[i] = loss_forward_pass(input_tensor=output_tensor[i - 1], one_hot_encoded_true_labels=one_hot_true_labels_matrix, loss_type=layers[i][
                'LOSS_TYPE'])

    return output_tensor


def backward_pass_controller(output_tensor, one_hot_true_labels_matrix, layers, params):
    params_grad = {}
    for i in range(len(layers), 0, -1):
        if layers[i]['TYPE'] == "LOSS":
            params_grad[i] = loss_backward_pass(input_tensor=output_tensor[i - 1], one_hot_encoded_true_labels=one_hot_true_labels_matrix, loss_type=layers[i]['LOSS_TYPE'])

        elif layers[i]['TYPE'] == "OUTPUT":
            params_grad[i] = output_backward_pass(output_grad=params_grad[i + 1]['INPUT_GRAD'], params=params[i], input_tensor=output_tensor[i - 1])

        elif layers[i]['TYPE'] == "HIDDEN":
            params_grad[i] = hidden_backward_pass(output_grad=params_grad[i + 1]['INPUT_GRAD'], output_tensor=output_tensor[i], params=params[i], activation_function=layers[i]['ACTIVATION_FUNCTION'],
                                                  input_tensor=output_tensor[i - 1])

    return params_grad


def hidden_forward_pass(input_tensor, params, activation_function):
    W = params['W']
    b = params['b']
    assert np.shape(input_tensor)[1] == np.shape(W)[0], "Dimension mismatch between X and W in hidden layer"

    a = np.dot(input_tensor, W) + b
    global output

    if activation_function == "SIGMOID":

        output = 1/(1 + np.exp(-1*a))
    elif activation_function == "RELU":
        output = np.maximum(a, 0)
    elif activation_function == "TANH":
        output = (np.exp(2*a) - 1)/ (np.exp(2*a) + 1)

    return output


def output_forward_pass(input_tensor, params):
    W = params['W']
    b = params['b']

    assert np.shape(input_tensor)[1] == np.shape(W)[0], "Dimension mismatch between X and W in output layer"

    a = np.dot(input_tensor, W) + b

    # Overflow prevention trick
    for img in a:
        img -= np.max(img)

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
    batch_size = np.shape(input_tensor)[0]

    # Calculating mean cross entropy loss

    if loss_type == 'CE':
        offset = math.pow(10, -10)
        total_loss = []
        for prob, actual in zip(y_hat, one_hot_encoded_true_labels):
            total_loss += [-1 * (actual_elem * np.log(prob_elem + offset) + (1 - actual_elem) *
                                 np.log(1 - prob_elem + offset)) for prob_elem, actual_elem in zip(prob, actual)]

        avg_loss = np.sum(total_loss) / np.shape(input_tensor)[0]
        avg_loss = round(avg_loss, 4)

    return avg_loss


def loss_backward_pass(input_tensor, one_hot_encoded_true_labels, loss_type):
    if loss_type == "CE":
        assert np.shape(input_tensor) == np.shape(one_hot_encoded_true_labels), "softmax output and one-hot encoded matrix not of same shape!"

        return {'INPUT_GRAD': input_tensor - one_hot_encoded_true_labels}


def output_backward_pass(output_grad, params, input_tensor):

    W = params['W']
    b = params['b']

    batch_size = np.shape(input_tensor)[0]
    W_rows = np.shape(W)[0]
    W_cols = np.shape(W)[1]

    # grad_weight: (5,100,10)
    # gradient : (10,5)
    # input_tensor: (5,100)
    # output_grad

    W_grad = np.zeros((batch_size, W_rows, W_cols))

    for gradient, input_row, i in zip(output_grad, input_tensor, range(batch_size)):
        W_grad[i, :, :] = np.outer(input_row, gradient)

    running_grad = np.zeros((batch_size, W_rows, W_cols))

    for i, gradient in enumerate(output_grad):
        running_grad[i, :, :] = np.multiply(W, gradient)

    param_grad = {'W': W_grad, 'b': output_grad, 'INPUT_GRAD': running_grad}

    # assert np.shape(param_grad['W']) == np.shape(W), "weight gradient and weight matrix shapes mismatch in output layer"
    # assert np.shape(param_grad['b']) == np.shape(b), "bias gradient and bias matrix shapes mismatch in output layer"

    return param_grad


def hidden_backward_pass(output_grad, output_tensor, params, activation_function, input_tensor):
    W = params['W']
    b = params['b']
    batch_size = np.shape(input_tensor)[0]
    W_rows = np.shape(W)[0]
    W_cols = np.shape(W)[1]

    # input_tensor: 5,100
    # gradient: 5,100,10

    running_grad = np.zeros((np.shape(output_grad)[0], np.shape(output_grad)[1]))
    global temp
    if activation_function == "SIGMOID":
        temp = np.multiply(output_tensor, 1 - output_tensor)

    elif activation_function == "RELU":
        temp = np.ones(output_tensor.shape)
        negatives_indices = np.where(input_tensor < 0)
        temp[negatives_indices] = 0

    elif activation_function == "TANH":
        temp = 1 - np.multiply(output_tensor, output_tensor)

    combined_grad = np.zeros((output_grad.shape[0], output_grad.shape[1]))
    for row, grad, idx in zip(temp, output_grad, range(output_grad.shape[0])):
        summed_grad = np.sum(grad, axis=1)
        combined_grad[idx, :] = np.multiply(summed_grad, row)

    W_grad = np.zeros((batch_size, W_rows, W_cols))

    for gradient, input_row, i in zip(combined_grad, input_tensor, range(batch_size)):
        W_grad[i, :, :] = np.outer(input_row, gradient)

    running_grad = np.zeros((batch_size, W_rows, W_cols))

    for i, gradient in enumerate(running_grad):
        running_grad[i, :, :] = np.multiply(W, gradient)

    param_grad = {'W': W_grad, 'b': combined_grad, 'INPUT_GRAD': running_grad}

    return param_grad


def get_one_hot_encoding(true_labels, batch_size, class_count):

    true_labels = [int(i) for i in true_labels]
    one_hot_matrix = np.zeros((batch_size, class_count), dtype=int)
    one_hot_matrix[np.arange(batch_size), true_labels] = 1

    return one_hot_matrix


def get_classification_acc(true_labels, prob):
    x = []
    for i in range(np.shape(prob)[0]):
        if np.argmax(prob[i]) == true_labels[i]:
            x.append(1)

    return round(len(x)/float(np.shape(prob)[0]), 4)


def update_params(learning_rate, momentum, decay, params, params_grad):

    for i in range(1, len(params)+1):
        temp_w_grad = np.mean(params_grad[i]['W'], axis=0)
        temp_b_grad = np.mean(params_grad[i]['b'], axis=0).reshape(np.shape(params[i]['b']))

        params[i]['V_W'] = -1*params[i]['V_W']*momentum + temp_w_grad*learning_rate + params[i]['W'] * decay
        params[i]['V_b'] = -1*params[i]['V_b']*momentum + temp_b_grad*learning_rate
        params[i]['W'] = params[i]['W'] - params[i]['V_W']
        params[i]['b'] = params[i]['b'] - params[i]['V_b']

    return params
