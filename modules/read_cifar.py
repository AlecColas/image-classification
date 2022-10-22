import pickle
import random
import numpy as np


def read_cifar_batch(file_path: str):
    """Extracts CIFAR data from one batch and converts the (data) and (labels) to arrays of type (np.float32) and (np.int64) respectively.

    Args:
        file_path (str): path of the CIFAR batch file to read.

    Returns:
        data_in_float32 (np.ndarray(np.float64)) : images extracted from CIFAR data.
        labels_in_int64 (np.ndarray(np.int32))   : corresponding labels extracted from CIFAR data.
    """

    with open(file_path, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    data_in_bytes = dict[b"data"]
    labels_in_bytes = dict[b"labels"]

    data_in_float32 = np.asarray(data_in_bytes, np.float32)
    labels_in_int64 = np.asarray(labels_in_bytes, np.int64)

    return (data_in_float32, labels_in_int64)


def read_cifar(folder_path: str):
    """Extracts data contained in all CIFAR-10 data batches and returns the concatenated data and labels arrays.

    Args:
        folder_path (str): path of the directory containing CIFAR-10 data to extract and concatenate.

    Returns:
        concact_data (np.ndarray(np.float64)) : images extracted from all CIFAR-10 data batches.
        concat_labels (np.ndarray(np.int32))  : corresponding labels extracted from all CIFAR-10 data batches.
    """

    print('Extracting CIFAR-10 data')

    axis_of_concat = 0

    data_0 = read_cifar_batch(folder_path + '/test_batch')
    concat_data = data_0[0]
    concat_labels = data_0[1]

    for i in range(1, 6):
        data_i = read_cifar_batch(folder_path + '/data_batch_' + str(i))

        concat_data = np.concatenate(
            (concat_data, data_i[0]), axis_of_concat)
        concat_labels = np.concatenate(
            (concat_labels, data_i[1]), axis_of_concat)

    return (concat_data, concat_labels)


def split_dataset(data, labels, split: float):
    """Splits CIFAR-10 concatenated dataset into a training set and a test set according to a given split_factor.
    Splitting is done randomly so that two successive calls shouldn't give the same output.
    CIFAR-10 dataset are partitioned in data and corresponding labels arrays.

    Args:
        data (np.ndarray(np.float32)): images extracted from all CIFAR-10 data batches.
        labels (np.ndarray(np.int64)): corresponding labels extracted from all CIFAR-10 data batches.
        split (float): The split factor used to split CIFAR-10 data in training and test data.
                        example : with split_factor = 0.8 : 80% data will be training data, and 20% will be test data.

    Returns:
        train_data (np.ndarray(np.float32)): the training data, and
        train_labels (np.ndarray(np.int64)): the corresponding labels.
        test_data (np.ndarray(np.float32)): the test data, and
        test_labels (np.ndarray(np.int64)): the corresponding labels.
    """

    print('Splitting dataset')

    nb_rows = len(data)

    train_data_length = int(nb_rows * split)
    test_data_length = nb_rows - train_data_length

    random_row_indices = random.sample(range(nb_rows), test_data_length)

    test_data = data[random_row_indices, :]
    test_labels = labels[random_row_indices]

    train_data = np.delete(data, random_row_indices, axis=0)
    train_labels = np.delete(labels, random_row_indices, axis=0)

    return (train_data, train_labels, test_data, test_labels)
