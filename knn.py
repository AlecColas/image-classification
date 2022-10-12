import numpy as np
from scipy import stats


def distance_matrix(a, b):
    # calculer la distance sur chaque pixel et chaque couleur
    diff = np.sqrt(a*a + b*b - 2*a*b)
    # faire la somme matriciel des couelurs RGB sur chaque pixel et chaque image
    somm = np.sum(diff, (1))
    # prendre la racine de la somme
    dists = np.sqrt(somm)
    return dists


def knn_predict(dists, labels_train, k=1):
    if (k <= 0 or k > np.shape(dists)[0]):
        return

    indexes_of_knn = np.argsort(dists, 0)[:k]
    labels_of_knn = labels_train[indexes_of_knn]
    return labels_of_knn


def classify_with_mode(labels_of_knn):
    return stats.mode(labels_of_knn, 0, keepdims=False).mode


def compute_accuracy(labels_test, computed_labels):
    nb_labels = len(labels_test)
    nb_well_classified = np.count_nonzero(labels_test == computed_labels)
    accuracy = nb_well_classified / nb_labels
    return accuracy


def evaluate_knn(data_train, labels_train, data_test, labels_test, k):
    L_data_train = np.shape(data_train)[0]

    if (k <= 0 or k > L_data_train):
        return

    computed_labels = []

    for i in range(len(data_test)):
        matrix_test_i = np.array([data_test[i], ] * L_data_train)

        dists = distance_matrix(data_train, matrix_test_i)
        labels_of_knn = knn_predict(dists, labels_train, k)

        computed_classification_i = classify_with_mode(labels_of_knn)
        computed_labels.append(computed_classification_i)

    return compute_accuracy(labels_test, computed_labels)
