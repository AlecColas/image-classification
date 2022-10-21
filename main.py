from modules.knn import evaluate_knn, evaluate_knn_optimized
from modules.read_cifar import read_cifar, split_dataset
from modules.helper import choose_classification_method, choose_split_factor, plot_and_save_fig

chosen_method = choose_classification_method()
split_factor = choose_split_factor()


dict = read_cifar('data')
(data, labels) = dict

data_train, labels_train, data_test, labels_test = split_dataset(
    data, labels, split_factor)

# accuracies = []
# list_k = list(range(1, 21, 1))
# print(list_k)
# for k in list_k:
#     print(k)
#     k_accuracy = evaluate_knn(data_train, labels_train,
#                               data_test, labels_test, k)
#     accuracies.append(k_accuracy)

# print(accuracies)

k_max = 20
accuracy_for_all_k = evaluate_knn_optimized(data_train, labels_train,
                                            data_test, labels_test, k_max)

plot_and_save_fig(k_max, accuracy_for_all_k, split_factor)
