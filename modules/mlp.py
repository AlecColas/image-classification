import numpy as np

MAX_LABEL = 9
MIN_LABEL = 0


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)


def one_hot(labels):
    nb_rows = labels.size

    hot_labels = np.zeros((nb_rows, MAX_LABEL+1))
    hot_labels[np.arange(nb_rows), labels] = 1

    return hot_labels


def learn_once_mse(w1, b1, w2, b2, data, targets, learning_rate):
    # resize targets
    targets = np.resize(targets, (len(targets), 1))

    # Forward pass
    a0 = data  # the data are the input of the first layer
    z1 = np.matmul(a0, w1) + b1  # input of the hidden layer
    # output of the hidden layer (sigmoid activation function)
    a1 = 1 / (1 + np.exp(-z1))
    z2 = np.matmul(a1, w2) + b2  # input of the output layer
    # output of the output layer (sigmoid activation function)
    a2 = 1 / (1 + np.exp(-z2))
    predictions = a2  # the predicted values are the outputs of the output layer

    # Compute loss (MSE)
    loss = np.mean(np.square(predictions - targets))
    print('MSE Loss before learning :', loss)

    # Gradient computation
    dC_da2 = 2*(a2 - targets)
    dC_dz2 = dC_da2 * (a2-np.square(a2))
    dC_dw2 = np.matmul(a1.T, dC_dz2)
    dC_db2 = np.sum(dC_dz2, axis=0, keepdims=True)

    dC_da1 = np.matmul(dC_dz2, w2.T)
    dC_dz1 = dC_da1 * (a1 - np.square(a1))
    dC_dw1 = np.matmul(a0.T, dC_dz1)
    dC_db1 = np.sum(dC_dz1, axis=0, keepdims=True)

    w1 += learning_rate * dC_dw1
    b1 += learning_rate * dC_db1
    w2 += learning_rate * dC_dw2
    b2 += learning_rate * dC_db2

    return (w1, b1, w2, b2, loss)


def learn_once_cross_entropy(w1, b1, w2, b2, data, targets, learning_rate):
    # resize targets
    targets = np.resize(targets, (len(targets), 1))

    # Forward pass
    a0 = data  # the data are the input of the first layer
    z1 = np.matmul(a0, w1) + b1  # input of the hidden layer
    # output of the hidden layer (sigmoid activation function)
    a1 = 1 / (1 + np.exp(-z1))
    z2 = np.matmul(a1, w2) + b2  # input of the output layer
    # output of the output layer (softmax activation function)
    a2 = softmax(z2)
    predictions = a2  # the predicted values are the outputs of the output layer

    targets_one_hot = one_hot(targets)
    # Compute loss (Cross Entropy)
    loss = - np.mean(targets_one_hot * np.log(predictions))
    print('Loss before learning with cross entropy :', loss)

    # Gradient computation
    ''' We admit that $`\frac{partial C}{partial Z^{(2)}} = A^{(2)} - Y`$. 
    Where $`Y`$ is a one-hot vector encoding the label. '''
    dC_dz2 = np.square(a2) - targets
    dC_dw2 = np.matmul(a1.T, dC_dz2)
    dC_db2 = np.sum(dC_dz2, axis=0, keepdims=True)

    dC_da1 = np.matmul(dC_dz2, w2.T)
    dC_dz1 = dC_da1 * (a1 - np.square(a1))
    dC_dw1 = np.matmul(a0.T, dC_dz1)
    dC_db1 = np.sum(dC_dz1, axis=0, keepdims=True)

    w1 += learning_rate * dC_dw1
    b1 += learning_rate * dC_db1
    w2 += learning_rate * dC_dw2
    b2 += learning_rate * dC_db2

    return (w1, b1, w2, b2, loss)


def train_mlp(w1, b1, w2, b2, data_train, labels_train, learning_rate, num_epoch):
    train_accuracies = np.zeros((num_epoch, 1))

    for k in range(num_epoch):
        print('Training MLP for epoch number :', k)
        (w1, b1, w2, b2, loss) = learn_once_cross_entropy(
            w1, b1, w2, b2, data_train, labels_train, learning_rate)

        train_accuracies[k] = loss

    return (w1, b1, w2, b2, train_accuracies)


def test_mlp(w1, b1, w2, b2, data_test, labels_test):

    # Forward pass
    a0 = data_test  # the data are the input of the first layer
    z1 = np.matmul(a0, w1) + b1  # input of the hidden layer
    # output of the hidden layer (sigmoid activation function)
    a1 = 1 / (1 + np.exp(-z1))
    z2 = np.matmul(a1, w2) + b2  # input of the output layer
    # output of the output layer (sigmoid activation function)
    a2 = 1 / (1 + np.exp(-z2))
    predictions = a2  # the predicted values are the outputs of the output layer

    nb_labels = len(labels_test)
    nb_well_classified = np.count_nonzero(predictions == labels_test)
    accuracy = nb_well_classified / nb_labels

    return accuracy


def run_mlp_training(data_train, labels_train, data_test, labels_test, d_h, learning_rate, num_epoch):
    d_in = np.shape(data_train)[1]
    d_out = 1

    # Random initialization of the network weights and biaises
    w1 = 2 * np.random.rand(d_in, d_h) - 1  # first layer weights
    b1 = np.zeros((1, d_h))  # first layer biaises
    w2 = 2 * np.random.rand(d_h, d_out) - 1  # second layer weights
    b2 = np.zeros((1, d_out))  # second layer biaises

    (w1, b1, w2, b2, train_accuracies) = train_mlp(w1, b1, w2,
                                                   b2, data_train, labels_train, learning_rate, num_epoch)

    final_accuracy = test_mlp(w1, b1, w2, b2, data_test, labels_test)

    return (train_accuracies, final_accuracy)
