import numpy as np
import pytest
from numpy import random

from modules.read_cifar import read_cifar, read_cifar_batch, split_dataset


def test_read_cifar_batch():

    int = random.randint(1, 7)
    if int == 6:
        data_in_float32, labels_in_int64 = read_cifar_batch("data/test_batch")
    else:
        data_in_float32, labels_in_int64 = read_cifar_batch(
            "data/data_batch_" + str(int)
        )

    assert type(data_in_float32) == np.ndarray
    assert type(labels_in_int64) == np.ndarray
    assert data_in_float32.dtype == np.float32
    assert labels_in_int64.dtype == np.int64
    assert np.shape(data_in_float32) == (10000, 3072)
    assert np.shape(labels_in_int64) == (10000,)

    return


def test_wrong_filepath():
    with pytest.raises(FileNotFoundError):
        read_cifar_batch("data_batch_0")
    return


def test_read_cifar():
    concat_batch, concat_labels = read_cifar("data")

    assert type(concat_batch) == np.ndarray
    assert type(concat_labels) == np.ndarray
    assert concat_batch.dtype == np.float32
    assert concat_labels.dtype == np.int64
    assert np.shape(concat_batch) == (60000, 3072)
    assert np.shape(concat_labels) == (60000,)

    return


def test_wrong_directory():
    with pytest.raises(FileNotFoundError):
        read_cifar("toto")
    return


def test_split_dataset():
    concat_batch, concat_labels = read_cifar("data")

    split = np.round(random.rand(), 2)

    train_data, train_labels, test_data, test_labels = split_dataset(
        concat_batch, concat_labels, split
    )

    assert type(train_data) == np.ndarray
    assert type(train_labels) == np.ndarray
    assert type(test_data) == np.ndarray
    assert type(test_labels) == np.ndarray

    assert train_data.dtype == np.float32
    assert train_labels.dtype == np.int64
    assert test_data.dtype == np.float32
    assert test_labels.dtype == np.int64

    L_train_data, l_train_data = np.shape(train_data)
    L_test_data, l_test_data = np.shape(test_data)

    L_train_labels = np.shape(train_labels)[0]
    L_test_labels = np.shape(test_labels)[0]

    assert L_train_data + L_test_data == 60000
    assert l_train_data == l_test_data == 3072
    assert L_train_labels + L_test_labels == 60000

    return
