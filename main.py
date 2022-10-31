"""Main script.

When launched, it guides the user from choosing a classification method and its parameters to plotting and saving the data.
"""

from matplotlib import pyplot as plt

from knn import evaluate_knn, evaluate_knn_optimized
from mlp import run_mlp_training
from read_cifar import read_cifar, split_dataset


def plot_and_save_fig(x_max, accuracies, split_factor, name):
    """Plot the accuracy of a classification method.

    When using KNN, it will plot the accuracy over k (he number of nearest neighbors used for classification).
    When using NN, it will plot the accuracy over the number of epochs spent in training.

    Args:
        x_max (int): The maximum number of neighbors used to evaluate the classification method.
        accuracies (List[int]): The computed accuracies of KNN method for k in range (1,k_max).
        split_factor (float): The split factor used to split CIFAR-10 data in training and test data.
    """
    print("Plotting figure")

    range_k_max = range(1, x_max + 1, 1)
    list_k = list(range_k_max)

    fig = plt.figure()
    plt.plot(
        list_k,
        accuracies,
        marker="o",
        linestyle="--",
        color="b",
        label="split_factor = " + str(split_factor),
    )

    plt.title(name + " method accuracy using CIFAR-10 data")
    plt.grid(True, which="both")
    plt.ylabel("Accuracy")
    plt.legend()

    if name == "knn":
        default_x_ticks = range_k_max
        plt.xticks(default_x_ticks, list_k)
        plt.xlabel("k number of neighbors")
    else:
        plt.xlabel("Epoch number")
    plt.show()

    save = 1
    if save == 1:
        fig.savefig("results/" + name + ".png")

    return


chosen_method = 2
split_factor = 0.9
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
