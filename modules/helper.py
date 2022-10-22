from matplotlib import pyplot as plt
from datetime import datetime


def choose_classification_method():
    """This function is a helper. It can be used to choose the classification method to evaluate.

    Returns:
        int: Representing th chosen method : 
        - 0 for unoptimized KNN method
        - 1 for optimized KNN
        - 2 for Neural Networks
    """

    method_int = 0
    method_int_test = False

    while not method_int_test:
        str_method_int = input(
            '\nMETHOD :\nChoose the method you want to use to classify CIFAR-10 images :  0 - KNN (unoptimized), 1 - KNN (optimized)\nYou entered : ')

        try:
            method_int = int(str_method_int)
            if (type(method_int) == int) and (0 <= method_int <= 1):
                method_int_test = True

            else:
                print('You must choose a method among : 0 - KNN (unoptimized), 1 - KNN (optimized)\n',
                      method_int, 'is invalid')
                method_int_test = False
        except Exception as e:
            method_int_test = False
            print(e)
    return method_int


def choose_split_factor():
    """This function is a helper. It can be used to choose the split factor used to split CIFAR-10 data in training and test data.

    Returns:
        float: the split factor between 0. and 1. The closer it is to 1, the smaller test data will be.
            Example : with split_factor = 0.8 : 80% data will be training data, and 20% will be test data.
    """

    split_factor = 1
    split_test = False

    while not (split_test):
        str_split_factor = input(
            '\nSPLIT FACTOR :\nEnter a float between 0 and 1 which determines the split factor between training and test sets.\nYou entered : ')

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


def choose_to_save():
    """This function is a helper. It can be used to choose whether to save the plotted figure or not.

    Returns:
        int: 0 if the user decided to not save the figure, 1 if he decided to save it.
    """

    save = 'n'
    save_test = False

    while not save_test:
        save = input(
            '\nSAVE PLOT :\nDo you want to save the current plot : type [y/yes] for yes and [n/no] for no.\nYou entered : ')

        try:
            if (save in ['y', 'yes', 'n', 'no']):
                save_test = True

            else:
                print('You must enter [y/yes] or [n/no] to choose :',
                      save, 'is invalid')
                save_test = False
        except Exception as e:
            save_test = False
            print(e)

    if (save in ['n', 'no']):
        return 0
    else:
        return 1


def plot_and_save_fig(k_max, accuracies, split_factor):
    """This function is a helper. It plots the accuracy of KNN method as a function of K with matplotlib and save the figure if wanted.

    Args:
        k_max (int): The maximum number of neighbors used to evaluate the classification method.
        accuracies (List[int]): The computed accuracies of KNN method for k in range (1,k_max).
        split_factor (float): The split factor used to split CIFAR-10 data in training and test data.
    """

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
    plt.show()

    save = choose_to_save()
    if (save == 1):
        fig.savefig('results/knn'+str(datetime.now())+'.png')

    return
