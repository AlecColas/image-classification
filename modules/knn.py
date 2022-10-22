import numpy as np
from scipy import stats


def distance_matrix(train, test):
    """Returns the L2 Euclidean distance matrix between two matrixes of shape (n,m) and (m,p).

    Args:
        train (np.ndarray(np.float32)): The training images data matrix of shape (n,m).
        test (np.ndarray(np.float32)): The test images data matrix of shape (m,p).

    Returns:
        np.ndarray(np.float32): The L2 Euclidean distance matrix between train and test, of shape (n,p).

    """

    print('Computing distance matrix between train and test sets')

    train2 = train*train
    test2 = test*test

    train2_sum = np.sum(train2, axis=1, keepdims=True)
    test2_sum = np.sum(test2, axis=1, keepdims=True)

    product = -2 * np.matmul(train, test.T)

    dists = np.sqrt(product + train2_sum + test2_sum.T)
    return dists


def knn_predict(dists, labels_train, k=1):
    """Computes the k-nearest neighbors labels for each image of data_test using distances between training data and test data.
    The distance minima are sorted along the y-axis, each column of dists corresponding to the distances between one test image and all training images.

    Args:
        dists (np.ndarray(np.float32)): the distance matrix between the train set and the test set.
        labels_train (np.ndarray(np.int64)): the corresponding labels of training data.
        k (int, optional): the chosen number of neighbors. Defaults to 1.

    Returns:
        np.ndarray(np.int64): the predicted labels for the images of data_test.
    """

    print('Extracting labels of k nearest neighbors')

    if (k <= 0 or k > np.shape(dists)[0]):
        return np.array([])

    indexes_of_knn = np.argsort(dists, axis=0)[0:k, :]

    labels_of_knn = labels_train[indexes_of_knn]
    return labels_of_knn


def classify_with_mode(labels_of_knn):
    """Computes classification of test images according to the most common label of their k-nearest neighbors in training data.

    Args:
        labels_of_knn (np.ndarray(np.int64)): an array of the computed possible labels for the images of data_test.
        Each column corresponds to the k labels of the nearest images to one test image.

    Returns:
        np.ndarray(np.int64): a list of the most common labels in the passed array along to the y-axis.
    """

    return stats.mode(labels_of_knn, axis=0, keepdims=False).mode


def compute_accuracy(labels_test, computed_labels) -> float:
    """Computes the classification rate (accuracy) of a method by comparing computed labels to true labels of test data.

    Args:
        labels_test (np.ndarray(np.int64)): the true labels of test images.
        computed_labels (np.ndarray(np.int64)): the computed labels of test images.

    Returns:
        float: the computed accuracy.
    """

    nb_labels = len(labels_test)
    nb_well_classified = np.count_nonzero(labels_test == computed_labels)
    accuracy = nb_well_classified / nb_labels
    return accuracy


def evaluate_knn(data_train, labels_train, data_test, labels_test, k: int):
    """Evaluates the classification rate of one knn method given k.

    Args:
        data_train (np.ndarray(np.float32)): the training data, and
        labels_train (np.ndarray(np.int64)): the corresponding labels.
        data_test (np.ndarray(np.float32)): the test data, and 
        labels_test (np.ndarray(np.int64)): the corresponding labels.
        k (int): the chosen number of neighbors.

    Returns:
        float: the classification rate (accuracy) of this k-nn method.
    """

    L_data_train = np.shape(data_train)[0]

    if (k <= 0 or k > L_data_train):
        return

    dists = distance_matrix(data_train, data_test)
    labels_of_knn = knn_predict(dists, labels_train, k)

    computed_labels_for_test_images = classify_with_mode(labels_of_knn)

    return compute_accuracy(labels_test, computed_labels_for_test_images)


def compute_accuracy_for_range_k(labels_test, computed_labels_for_k_max, k_max):
    """Computes the classification rate (accuracy) of a knn methods for k in range(1, k_max) by comparing computed labels to true labels of test data.

    Args:
        labels_test (np.ndarray(np.int64)): the true labels of test images.
        computed_labels_for_all_k (np.ndarray(np.int64)): the computed labels of test images for k = k_max
        k_max (int): the chosen maximum number of neighbors.

    Returns:
        list(float): the list of accuracies for knn methods for k in range (1, k_max)
    """

    print('Computing accuracies')
    nb_labels = len(labels_test)

    # list of accuracies for all k in range (1, k_max)
    accuracies = []

    for k in range(k_max):
        # extract column k representing the computed labels for data_test for k
        computed_labels_for_k = computed_labels_for_k_max[k]

        # get the number of well computed labels for k-neighbors
        nb_well_classified_for_k = np.count_nonzero(
            labels_test == computed_labels_for_k)
        # compute the accuracy for k-neighbors
        accuracy_for_k = nb_well_classified_for_k / nb_labels
        # insert accuracy in list
        accuracies.append(accuracy_for_k)

    return accuracies


def evaluate_knn_optimized(data_train, labels_train, data_test, labels_test, k_max: int):
    """An optimized version of evaluate_knn() that computes the classification rate of knn methods for k in range(1, k_max).

    Args:
        data_train (np.ndarray(np.float32)): the training data, and
        labels_train (np.ndarray(np.int64)): the corresponding labels.
        data_test (np.ndarray(np.float32)): the test data, and 
        labels_test (np.ndarray(np.int64)): the corresponding labels.
        k_max (int): the chosen maximum number of neighbors.

    Returns:
        list(float): the accuracy list of knn methods for k in range(1, k_max).
    """

    print('Starting knn evaluation')

    # check if k_max value is correct
    L_data_train = np.shape(data_train)[0]
    if (k_max <= 0 or k_max > L_data_train):
        return

    dists = distance_matrix(data_train, data_test)
    # get the computed labels for data_test, k = k_max
    computed_labels_for_k_max_neighbors = knn_predict(
        dists, labels_train, k_max)

    computed_labels_for_test_images = []
    for i in range(1, k_max+1):
        computed_labels_for_test_images_for_k = classify_with_mode(
            computed_labels_for_k_max_neighbors[:i, :])

        computed_labels_for_test_images.append(
            computed_labels_for_test_images_for_k)

    # compute the accuracy of the knn methods for all k in range (1, k_max)
    accuracy_for_all_k = compute_accuracy_for_range_k(
        labels_test, computed_labels_for_test_images, k_max)

    return accuracy_for_all_k
