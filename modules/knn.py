import numpy as np
from scipy import stats


def distance_matrix(train, test):
    # returns a matrix of size len(train) x len(test)
    train2 = train*train
    test2 = test*test

    train2_sum = np.sum(train2, axis=1, keepdims=True)
    test2_sum = np.sum(test2, axis=1, keepdims=True)

    product = -2 * np.matmul(train, test.T)

    dists = np.sqrt(product + train2_sum + test2_sum.T)
    return dists


def knn_predict(dists, labels_train, k=1):
    if (k <= 0 or k > np.shape(dists)[0]):
        return np.array([])

    indexes_of_knn = np.argsort(dists, axis=0)[0:k, :]

    labels_of_knn = labels_train[indexes_of_knn]
    return labels_of_knn


def classify_with_mode(labels_of_knn):
    return stats.mode(labels_of_knn, axis=0, keepdims=False).mode


def compute_accuracy(labels_test, computed_labels) -> float:
    nb_labels = len(labels_test)
    nb_well_classified = np.count_nonzero(labels_test == computed_labels)
    accuracy = nb_well_classified / nb_labels
    return accuracy


def evaluate_knn(data_train, labels_train, data_test, labels_test, k: int):
    L_data_train = np.shape(data_train)[0]

    if (k <= 0 or k > L_data_train):
        return

    dists = distance_matrix(data_train, data_test)
    labels_of_knn = knn_predict(dists, labels_train, k)

    computed_labels_for_test_images = classify_with_mode(labels_of_knn)

    return compute_accuracy(labels_test, computed_labels_for_test_images)


def compute_accuracy_for_range_k(labels_test, computed_labels_for_all_k, k_max):
    nb_labels = len(labels_test)

    # list of accuracies for all k in range (1, k_max)
    accuracy = []

    for k in range(k_max):
        # extract column k representing the computed labels for data_test for k
        computed_labels_for_k = computed_labels_for_all_k[k]

        # get the number of well computed labels for k-neighbors
        nb_well_classified = np.count_nonzero(
            labels_test == computed_labels_for_k)
        # compute the accuracy for k-neighbors
        accuracy_for_k = nb_well_classified / nb_labels
        # insert accuracy in list
        accuracy.append(accuracy_for_k)

    return accuracy


def evaluate_knn_optimized(data_train, labels_train, data_test, labels_test, k_max: int):

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

    # compute the accuracy of the knn method for all k in range (1, k_max)
    accuracy_for_all_k = compute_accuracy_for_range_k(
        labels_test, computed_labels_for_test_images, k_max)

    return accuracy_for_all_k
