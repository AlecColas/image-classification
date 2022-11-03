"""Functions to perform classification with Artificial Neural Networks.

The entry point is the function run_mlp_training().
You can either choose to use Mean Square Error (MSE) or Cross Entropy as the error function in train_mlp().
"""

import numpy as np


def softmax(x):
    """Compute the softmax values for each sets (rows) of scores in x.

    Parameters
    ----------
    x : np.ndarray(np.float32)
        A 2-D array containing the correspondence scores between one image and each labels (coded in rows).

    Returns
    -------
    np.ndarray(np.float32)
        The softmax activated scores.
    """
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps, axis=1, keepdims=True)


def one_hot(labels):
    """Return the one-hot matrix of label.

    Parameters
    ----------
    labels : np.ndarray(np.int64)
        An (n)-D array of labels.

    Returns
    -------
    np.ndarray(np.float64)
        The corresponding (n+1)-D one-hot matrix.
    """
    dimensions = np.max(labels) + 1
    hot_labels = np.eye(dimensions)[labels]

    return hot_labels


def learn_once_mse(w1, b1, w2, b2, data, targets, learning_rate):
    """Perform one gradient descent step using the Mean Square Error (MSE).

    Parameters
    ----------
    w1 : np.ndarray(np.float32)
        The weight matrix (d_in x d_h) of the first (hidden) layer.
    b1 : np.ndarray(np.float32)
        The bias matrix (1, d_h) of the first (hidden) layer.
    w2 : np.ndarray(np.float32)
        The weight matrix (d_h x d_out) of the output layer.
    b2 : np.ndarray(np.float32)
        The bias matrix (1 x d_out) of the output layer.
    data : np.ndarray(np.float32)
        The input matrix (batch_size x d_in) containing training images in rows, and
    targets : np.ndarray(np.int64)
        the vector (batch_size) of corresponding labels for each image.
    learning_rate : float
        The learning rate.

    Returns
    -------
    w1 : np.ndarray(np.float32)
        The updated weight matrix (d_in x d_h) of the first (hidden) layer.
    b1 : np.ndarray(np.float32)
        The updated bias matrix (1, d_h) of the first (hidden) layer.
    w2 : np.ndarray(np.float32)
        The updated weight matrix (d_h x d_out) of the output layer.
    b2 : np.ndarray(np.float32)
        The updated bias matrix (1 x d_out) of the output layer.
    loss : float
        The value of the loss for one training epoch.
    """
    N = data.shape[0]

    # Resize targets from tuple to np.ndarray
    r_targets = np.array([targets]).T

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
    loss = np.mean(np.square(predictions - r_targets))
    print("MSE Loss before learning :", loss)

    # Gradient computation
    dC_da2 = 2 * (a2 - r_targets) / N
    dC_dz2 = dC_da2 * (a2 - np.square(a2))
    dC_dw2 = np.matmul(a1.T, dC_dz2)
    dC_db2 = np.sum(dC_dz2, axis=0, keepdims=True) / N

    dC_da1 = np.matmul(dC_dz2, w2.T)
    dC_dz1 = dC_da1 * (a1 - np.square(a1))
    dC_dw1 = np.matmul(a0.T, dC_dz1)
    dC_db1 = np.sum(dC_dz1, axis=0, keepdims=True) / N

    w1 -= learning_rate * dC_dw1
    b1 -= learning_rate * dC_db1
    w2 -= learning_rate * dC_dw2
    b2 -= learning_rate * dC_db2

    return (w1, b1, w2, b2, loss)


def learn_once_cross_entropy(w1, b1, w2, b2, data, labels_train, learning_rate):
    """Perform one gradient descent step using Cross Entropy (softmax activation function + one hot encoding of labels).

    Parameters
    ----------
    w1 : np.ndarray(np.float32)
        The weight matrix (d_in x d_h) of the first (hidden) layer.
    b1 : np.ndarray(np.float32)
        The bias matrix (1, d_h) of the first (hidden) layer.
    w2 : np.ndarray(np.float32)
        The weight matrix (d_h x d_out) of the output layer.
    b2 : np.ndarray(np.float32)
        The bias matrix (1 x d_out) of the output layer.
    data : np.ndarray(np.float32)
        The input matrix (batch_size x d_in) containing training images in rows, and
    labels_train : np.ndarray(np.int64)
        the vector (batch_size) of corresponding labels for each image.
    learning_rate : float
        The learning rate.

    Returns
    -------
    w1 : np.ndarray(np.float32)
        The updated weight matrix (d_in x d_h) of the first (hidden) layer.
    b1 : np.ndarray(np.float32)
        The updated bias matrix (1, d_h) of the first (hidden) layer.
    w2 : np.ndarray(np.float32)
        The updated weight matrix (d_h x d_out) of the output layer.
    b2 : np.ndarray(np.float32)
        The updated bias matrix (1 x d_out) of the output layer.
    loss : float
        The value of the loss for one training epoch.
    """
    N = data.shape[0]

    # Forward pass
    a0 = data  # the data are the input of the first layer
    z1 = np.matmul(a0, w1) + b1  # input of the hidden layer
    # output of the hidden layer (sigmoid activation function)
    a1 = 1 / (1 + np.exp(-z1))
    z2 = np.matmul(a1, w2) + b2  # input of the output layer
    # output of the output layer (softmax activation function)
    a2 = softmax(z2)
    predictions = a2  # the predicted values are the outputs of the output layer

    targets_one_hot = one_hot(labels_train)
    # Compute loss (Cross Entropy)
    loss = -np.mean(targets_one_hot * np.log(predictions))
    print("Loss before learning with cross entropy :", loss)

    # Gradient computation
    # We admit that $`\frac{partial C}{partial Z^{(2)}} = A^{(2)} - Y`$.
    dC_dz2 = a2 - targets_one_hot
    dC_dw2 = np.matmul(a1.T, dC_dz2) / N
    dC_db2 = np.sum(dC_dz2, axis=0, keepdims=True) / N

    dC_da1 = np.matmul(dC_dz2, w2.T)
    dC_dz1 = dC_da1 * (a1 - np.square(a1))
    dC_dw1 = np.matmul(a0.T, dC_dz1) / N
    dC_db1 = np.sum(dC_dz1, axis=0, keepdims=True) / N

    w1 -= learning_rate * dC_dw1
    b1 -= learning_rate * dC_db1
    w2 -= learning_rate * dC_dw2
    b2 -= learning_rate * dC_db2

    return (w1, b1, w2, b2, loss)


def train_mlp(w1, b1, w2, b2, data_train, labels_train, learning_rate, num_epochs):
    """Perform num_epoch training steps with given learning_rate and loss function.

    Parameters
    ----------
    w1 : np.ndarray(np.float32)
        The weight matrix (d_in x d_h) of the first (hidden) layer.
    b1 : np.ndarray(np.float32)
        The bias matrix (1, d_h) of the first (hidden) layer.
    w2 : np.ndarray(np.float32)
        The weight matrix (d_h x d_out) of the output layer.
    b2 : np.ndarray(np.float32)
        The bias matrix (1 x d_out) of the output layer.
    data_train : np.ndarray(np.float32)
        The input matrix (batch_size x d_in) containing training images in rows, and
    labels_train : np.ndarray(np.int64)
        The vector (batch_size) of corresponding labels for each training image.
    learning_rate : float
        The learning rate.
    num_epochs : int
        The number of epochs to perform training.

    Returns
    -------
    w1 : np.ndarray(np.float32)
        The updated weight matrix (d_in x d_h) of the first (hidden) layer after complete training.
    b1 : np.ndarray(np.float32)
        The updated bias matrix (1, d_h) of the first (hidden) layer after complete training.
    w2 : np.ndarray(np.float32)
        The updated weight matrix (d_h x d_out) of the output layer after complete training.
    b2 : np.ndarray(np.float32)
        The updated bias matrix (1 x d_out) of the output layer after complete training.
    train_accuracies : np.ndarray(float32)
        A vector containing the accuracy before training for each epoch.
    """
    train_accuracies = np.zeros((num_epochs, 1))

    for k in range(num_epochs):
        print("Training MLP for epoch number :", k)
        (w1, b1, w2, b2, loss) = learn_once_cross_entropy(
            w1, b1, w2, b2, data_train, labels_train, learning_rate
        )

        # Forward pass
        a0 = data_train  # the data are the input of the first layer
        z1 = np.matmul(a0, w1) + b1  # input of the hidden layer
        # output of the hidden layer (sigmoid activation function)
        a1 = 1 / (1 + np.exp(-z1))
        z2 = np.matmul(a1, w2) + b2  # input of the output layer
        # output of the output layer (softmax activation function): the predicted values are the outputs of the output layer
        predictions = softmax(z2)

        # compute accuracy
        classification = np.argmax(predictions, axis=1, keepdims=True)
        accuracy = np.count_nonzero(classification[:, 0] == labels_train) / len(
            labels_train
        )
        print("Accuracy is : ", accuracy)

        train_accuracies[k] = accuracy

    return (w1, b1, w2, b2, train_accuracies)


def test_mlp(w1, b1, w2, b2, data_test, labels_test):
    """Test the trained network on the test set.

    Parameters
    ----------
    w1 : np.ndarray(np.float32)
        The weight matrix (d_in x d_h) of the first (hidden) layer.
    b1 : np.ndarray(np.float32)
        The bias matrix (1, d_h) of the first (hidden) layer.
    w2 : np.ndarray(np.float32)
        The weight matrix (d_h x d_out) of the output layer.
    b2 : np.ndarray(np.float32)
        The bias matrix (1 x d_out) of the output layer.
    data_test : np.ndarray(np.float32)
        The input matrix (batch_size x d_in) containing test images in rows, and
    labels_test : np.ndarray(np.int64)
        the vector (batch_size) of corresponding labels for each image.

    Returns
    -------
    float
        The testing accuracy.
    """
    # Forward pass
    a0 = data_test  # the data are the input of the first layer
    z1 = np.matmul(a0, w1) + b1  # input of the hidden layer
    # output of the hidden layer (sigmoid activation function)
    a1 = 1 / (1 + np.exp(-z1))
    z2 = np.matmul(a1, w2) + b2  # input of the output layer
    # output of the output layer (sigmoid activation function)
    a2 = 1 / (1 + np.exp(-z2))
    predictions = a2  # the predicted values are the outputs of the output layer

    classification = np.argmax(predictions, axis=1, keepdims=True)

    nb_labels = len(labels_test)
    nb_well_classified = np.count_nonzero(classification[:, 0] == labels_test)
    accuracy = nb_well_classified / nb_labels

    return accuracy


def run_mlp_training(
    data_train, labels_train, data_test, labels_test, d_h, learning_rate, num_epochs
):
    """Train an MLP classifier and test the trained network (weights and biases) on the test set.

    Parameters
    ----------
    data_train : np.ndarray(np.float32)
        The input matrix (batch_size x d_in) containing training images in rows, and
    labels_train : np.ndarray(np.int64)
        The vector (batch_size) of corresponding labels for each training image.
    data_test : np.ndarray(np.float32)
        The input matrix (batch_size x d_in) containing test images in rows, and
    labels_test : np.ndarray(np.int64)
        the vector (batch_size) of corresponding labels for each image.
    d_h : int
        The number of neurons in the hidden layer.
    learning_rate : float
        The learning rate.
    num_epochs : int
        The number of epochs to perform training.

    Returns
    -------
    train_accuracies : np.ndarray(float32)
        A vector containing the accuracy before training for each epoch.
    final_accuracy : float
        The testing accuracy.
    """
    d_in = np.shape(data_train)[1]
    d_out = 10

    # Random initialization of the network weights and biaises
    w1 = 2 * np.random.rand(d_in, d_h) - 1  # first layer weights
    b1 = np.zeros((1, d_h))  # first layer biaises
    w2 = 2 * np.random.rand(d_h, d_out) - 1  # second layer weights
    b2 = np.zeros((1, d_out))  # second layer biaises

    (w1, b1, w2, b2, train_accuracies) = train_mlp(
        w1, b1, w2, b2, data_train, labels_train, learning_rate, num_epochs
    )

    final_accuracy = test_mlp(w1, b1, w2, b2, data_test, labels_test)

    return (train_accuracies, final_accuracy)
