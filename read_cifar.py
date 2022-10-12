import pickle
import random
import numpy as np


def read_cifar_batch(file_path):
    with open(file_path, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    data_in_bytes = dict[b"data"]
    labels_in_bytes = dict[b"labels"]

    data_in_float32 = np.asarray(data_in_bytes, np.float32)
    labels_in_int64 = np.asarray(labels_in_bytes, np.int64)

    return (data_in_float32, labels_in_int64)


def read_cifar(folder_path):
    axis_of_concat = 0

    data_0 = read_cifar_batch(folder_path + '/test_batch')
    concat_batch = data_0[0]
    concat_labels = data_0[1]

    for i in range(1, 6):
        data_i = read_cifar_batch(folder_path + '/data_batch_' + str(i))

        concat_batch = np.concatenate(
            (concat_batch, data_i[0]), axis_of_concat)
        concat_labels = np.concatenate(
            (concat_labels, data_i[1]), axis_of_concat)

    return (concat_batch, concat_labels)


def split_dataset(data, labels, split):
    nb_rows = len(data)

    train_data_length = int(nb_rows * split)
    test_data_length = nb_rows - train_data_length

    random_row_indices = random.sample(range(nb_rows), test_data_length)

    test_data = data[random_row_indices, :]
    test_labels = labels[random_row_indices]

    train_data = np.delete(data, random_row_indices, axis=0)
    train_labels = np.delete(labels, random_row_indices, axis=0)

    return (train_data, train_labels, test_data, test_labels)


dict = read_cifar('data')

(data, labels) = dict

print('Length data : ', np.shape(data))
print('Length labels : ', np.shape(labels))

split_dataset(data, labels, 0.8)
