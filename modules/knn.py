import numpy as np
from scipy import stats


def distance_matrix(a, b):
    dists = np.linalg.norm(a - b, axis=1)
    return dists


def knn_predict(dists, labels_train, k=1):
    if (k <= 0 or k > np.shape(dists)[0]):
        return np.array([])

    indexes_of_knn = np.argsort(dists, 0)[:k]
    labels_of_knn = labels_train[indexes_of_knn]
    return labels_of_knn


def classify_with_mode(labels_of_knn):
    return stats.mode(labels_of_knn, 0, keepdims=False).mode


def compute_accuracy(labels_test, computed_labels) -> float:
    nb_labels = len(labels_test)
    nb_well_classified = np.count_nonzero(labels_test == computed_labels)
    accuracy = nb_well_classified / nb_labels
    return accuracy


def evaluate_knn(data_train, labels_train, data_test, labels_test, k: int):
    L_data_train = np.shape(data_train)[0]
    L_data_test = len(data_test)

    if (k <= 0 or k > L_data_train):
        return

    computed_labels = []

    for i in range(L_data_test):
        print('Computing distance matrix for sample ',
              str(i), 'over', str(L_data_test))
        matrix_test_i = np.array([data_test[i], ] * L_data_train)

        dists = distance_matrix(data_train, matrix_test_i)
        labels_of_knn = knn_predict(dists, labels_train, k)

        computed_classification_i = classify_with_mode(labels_of_knn)
        computed_labels.append(computed_classification_i)

    return compute_accuracy(labels_test, computed_labels)


def evaluate_classification_for_range_kmax(data_train, labels_train, data_test, k_max: int):

    # First we pick a test image in data_test
    # We get the labels of the k-nn for k = k_max
    # Thus we have all k-nn for k in range (1, k_max)
    # Then we compute the mode (nearest class' label) for all k in range (1,k_max) for this sample image

    # We reproduce this to all rows in data_test (each row representing 1 image)
    # Then we extract the computed labels, for each k in range (1, k_max+1), for each test image

    L_data_train = np.shape(data_train)[0]
    L_data_test = len(data_test)

    computed_labels_for_all_k = []

    for i in range(L_data_test):
        print('Computing distance matrix for sample ',
              str(i), 'over', str(L_data_test))

        matrix_test_i = np.array([data_test[i], ] * L_data_train)
        dists = distance_matrix(data_train, matrix_test_i)

        labels_of_knn = knn_predict(dists, labels_train, k_max)

        computed_classifications_for_test_i_and_all_k = []

        for j in range(1, k_max+1):
            computed_classifications_for_test_i_and_all_k.append(
                classify_with_mode(labels_of_knn[:j]))

        computed_labels_for_all_k.append(
            computed_classifications_for_test_i_and_all_k)

    return np.array(computed_labels_for_all_k)


def compute_accuracy_for_range_k(labels_test, computed_labels_for_all_k, k_max):
    nb_labels = len(labels_test)

    # list of accuracies for all k in range (1, k_max)
    accuracy = []

    for k in range(k_max):
        # extract column k representing the computed labels for data_test for k
        computed_labels_for_k = computed_labels_for_all_k[:, k]

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

    # get the computed labels for data_test, for all k in range (1,k_max)
    computed_labels_for_all_k = evaluate_classification_for_range_kmax(
        data_train, labels_train, data_test, k_max)

    # compute the accuracy of the knn method for all k in range (1, k_max)
    accuracy_for_all_k = compute_accuracy_for_range_k(
        labels_test, computed_labels_for_all_k, k_max)

    return accuracy_for_all_k
