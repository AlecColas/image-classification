from matplotlib import pyplot as plt
from modules.knn import evaluate_knn
from modules.read_cifar import read_cifar, split_dataset


split_test = False
split_factor = 1

while split_test != True:
    str_split_factor = input(
        'Enter a float between 0 and 1 which determines the split factor between training and test sets :  ')

    try:
        split_factor = float(str_split_factor)
        if (type(split_factor) == float) and (0. <= split_factor < 1.):
            split_test = True

        else:
            print('You must enter a float between 0 and 1 :',
                  split_factor, 'is invalid')
            split_test = False
    except Exception as e:
        split_test = False
        print(e)


dict = read_cifar('data')
(data, labels) = dict

data_train, labels_train, data_test, labels_test = split_dataset(
    data, labels, split_factor)

accuracies = []
list_k = list(range(1, 21, 1))
print(list_k)
for k in list_k:
    print(k)
    k_accuracy = evaluate_knn(data_train, labels_train,
                              data_test, labels_test, k)
    accuracies.append(k_accuracy)

print(accuracies)
plt.plot(list_k, accuracies, 'bo')
plt.show()
