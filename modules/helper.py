from matplotlib import pyplot as plt
from datetime import datetime


def choose_split_factor():
    split_factor = 1
    split_test = False

    while not (split_test):
        str_split_factor = input(
            'Enter a float between 0 and 1 which determines the split factor between training and test sets :  ')

        try:
            split_factor = float(str_split_factor)
            if (type(split_factor) == float) and (0. < split_factor < 1.):
                split_test = True

            else:
                print('You must enter a float between 0 and 1 :',
                      split_factor, 'is invalid')
                split_test = False
        except Exception as e:
            split_test = False
            print(e)
    return split_factor


def plot_and_save_fig(k_max, accuracies, split_factor):
    print('Plotting figure')

    range_k_max = range(1, k_max+1, 1)
    list_k = list(range_k_max)
    default_x_ticks = range_k_max

    fig = plt.figure()
    plt.plot(list_k, accuracies, marker='o',
             linestyle='--', color='b', label='split_factor = '+str(split_factor))

    plt.xticks(default_x_ticks, list_k)
    plt.grid(True, which='both')

    plt.xlabel('k number of neighbors')
    plt.ylabel('Accuracy')
    plt.title('KNN method accuracy using CIFAR-10 data')
    plt.legend()

    fig.savefig('results/knn'+str(datetime.now())+'.png')
    plt.show()

    return
