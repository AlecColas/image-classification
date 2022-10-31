"""Main script.

When launched, it guides the user from choosing a classification method and its parameters to plotting and saving the data.
"""

from helpers import choose_classification_method, choose_split_factor, plot_and_save_fig
from knn import evaluate_knn, evaluate_knn_optimized
from mlp import run_mlp_training
from read_cifar import read_cifar, split_dataset

chosen_method = choose_classification_method()
split_factor = choose_split_factor()
k_max = 20

dict = read_cifar("data")
(data, labels) = dict

data_train, labels_train, data_test, labels_test = split_dataset(
    data, labels, split_factor
)


if chosen_method == 0:
    accuracies = []
    list_k = list(range(1, 21, 1))
    for k in list_k:
        print("Evaluating KNN for k = ", k)
        k_accuracy = evaluate_knn(data_train, labels_train, data_test, labels_test, k)
        accuracies.append(k_accuracy)

    plot_and_save_fig(k_max, accuracies, split_factor, "knn")


elif chosen_method == 1:

    accuracy_for_all_k = evaluate_knn_optimized(
        data_train, labels_train, data_test, labels_test, k_max
    )

    plot_and_save_fig(k_max, accuracy_for_all_k, split_factor, "knn")

elif chosen_method == 2:
    num_epoch = 100
    (train_accuracies, train_losses, final_accuracy) = run_mlp_training(
        data_train, labels_train, data_test, labels_test, 64, 0.1, num_epoch
    )

    print("Final accuracy for test data is :", final_accuracy)

    plot_and_save_fig(num_epoch, train_accuracies, split_factor, "mlp")
    plot_and_save_fig(num_epoch, train_losses, split_factor, "mlp loss")
