# Image Classification

A comparison of two image classification methods in Python :

- k-nearest neighbors (KNN)
- and artificial neural networks (NN)

The data used to test these methods comes from the CIFAR-10 dataset : https://www.cs.toronto.edu/~kriz/cifar.html

This repository follows the instructions of this tutorial : https://gitlab.ec-lyon.fr/qgalloue/image_classification_instructions

## Installation

This project requires [python3](https://www.python.org/), and common libraries installed :

- [NumPy](https://numpy.org/) for computing with arrays
- [Pickle](https://docs.python.org/3/library/pickle.html) for Python object serialization
- [Pytest](https://docs.pytest.org/en/7.1.x/) for unit tests

This project follows the [PEP8](https://peps.python.org/pep-0008/) and [PEP257](https://peps.python.org/pep-0257/) Recommandations.

## Usage

Firstly, you need to read CIFAR-10 dataset, given that you have copied the data batches locally in a `data/` folder.
You can use this code to read and split the data in training and test sets :

```Python
from read_cifar import read_cifar, split_dataset

(data, labels) = read_cifar("data")

data_train, labels_train, data_test, labels_test = split_dataset(
    data, labels, split_factor
)
```

Thus, you can choose to use either KNN or Artificial Neural Network to perform classification :

For KNN :

```Python
from knn import evaluate_knn, evaluate_knn_optimized

# Classic version :
accuracy = evaluate_knn(data_train, labels_train, data_test, labels_test, k)

# Optimized version :
accuracies_for_range_k = evaluate_knn_optimized(
        data_train, labels_train, data_test, labels_test, k_max
    )

```

For Artifical Neural Network :

```Python
from mlp import run_mlp_training

(train_accuracies, train_losses, final_accuracy) = run_mlp_training(
        data_train, labels_train, data_test, labels_test, d_h, learning_rate, num_epoch
    )
```

## Project Architecture

    .
    ├── knn.py                 # KNN library file
    ├── mlp.py                 # Artificial Neural Network library file
    ├── read_cifar.py          # CIFAR reader library file
    ├── results                # Graphs
        ├── mlp.png
        └── knn.png
    └── tests                  # Unittests directory
        ├── test_knn.py
        ├── test_mlp.py
        └── test_read_cifar.py

Functions used for each classification method are contained in dedicated files.
Unit tests for each of these functions appear in the [tests/](tests/) directory under the name _test+\_[method_name].py_.

## CIFAR-10

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class.

The classes in the dataset, are : airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.
These classes are completely mutually exclusive.

## Support / Contributing

If you want to propose any improvements or need any help, feel free to contribute by [opening an issue](https://gitlab.ec-lyon.fr/colasa/image-classification/-/issues/new).

## References

- [Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf) by _Alex Krizhevsky_
- [Download CIFAR datasets](https://www.cs.toronto.edu/~kriz/cifar.html) by _Alex Krizhevsky_
- [Tutorial to Image Classification](https://gitlab.ec-lyon.fr/qgalloue/image_classification_instructions) by _Quentin Gallouédec_
- [What is the k-nearest neighbors method](https://www.ibm.com/topics/knn) by _IBM_
- [What are Neural Networks](https://www.ibm.com/cloud/learn/neural-networks) by _IBM_

## License

This project is licensed under the [MIT License](https://github.com/git/git-scm.com/blob/main/MIT-LICENSE.txt).

## Author

- [Alexandre Colas](https://gitlab.ec-lyon.fr/colasa)
