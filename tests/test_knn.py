import pytest
import numpy as np

from modules.knn import compute_accuracy, distance_matrix, evaluate_knn, knn_predict


a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = np.array([7, 8, 9])

a1 = np.array([a, a, a])
b1 = np.array([b, b])
c1 = np.array([c, c])


def test_distance_matrix():
    dists = distance_matrix(a1, b1)
    assert np.shape(dists) == (3, 2)
    assert np.array_equal(dists, [[np.sqrt(27), np.sqrt(27)], ]*3)

    dists1 = distance_matrix(a1, c1)
    assert np.shape(dists1) == (3, 2)
    assert np.array_equal(dists1, [[np.sqrt(108), np.sqrt(108)], ]*3)

    return


def test_incompatible_arrays_distance_matrix():
    d1 = np.array([[1, 2], [1, 2]])
    with pytest.raises(ValueError):
        distance_matrix(a1, d1)
    return


def test_predict_knn():
    predicts = np.array(['p1', 'p2', 'p3'])

    dists = distance_matrix(a1, b1)
    results = knn_predict(dists, predicts, 2)
    assert np.shape(results) == (2, 2)
    assert np.array_equal(results, np.array([['p1', 'p1'], ['p2', 'p2']]))

    dists1 = distance_matrix(a1, c1)
    results1 = knn_predict(dists1, predicts)
    assert np.shape(results1) == (1, 2)
    assert np.array_equal(results1, np.array([['p1', 'p1']]))

    assert len(knn_predict(dists, predicts, -1)) == 0
    assert len(knn_predict(dists, predicts, 4)) == 0

    return


def test_incompatible_arrays_knn_predict():
    predicts = np.array(['p1', 'p2'])
    dists = distance_matrix(a1, b1)

    with pytest.raises(IndexError):
        knn_predict(dists, predicts, 3)
    return


def test_compute_accuracy():
    predicts = np.array(['p1', 'p2', 'p3', 'p4'])
    true_values = np.array(['p1', 'p1', 'p3', 'p4'])
    assert compute_accuracy(true_values, predicts) == 0.75
    return


def test_incompatible_arrays_compute_accuracy():
    predicts = np.array(['p1', 'p2', 'p3'])
    true_values = np.array(['p1', 'p1'])
    with pytest.raises(ValueError):
        compute_accuracy(true_values, predicts)


def test_evaluate_knn():
    assert evaluate_knn([], [], [], [], -1) == None
    assert evaluate_knn([0, 1], [], [], [], 3) == None

    data_train = np.array([a, b, a, a])
    labels_train = np.array(['p1', 'p2', 'p1', 'p1'])
    data_test = np.array([a, a, b, a])
    labels_test = np.array(['p1', 'p1', 'p1', 'p3'])
    k = 1

    accuracy = evaluate_knn(data_train, labels_train,
                            data_test, labels_test, k)
    assert accuracy == 0.5

    k = 4
    accuracy = evaluate_knn(data_train, labels_train,
                            data_test, labels_test, k)
    assert accuracy == 0.75
    return
