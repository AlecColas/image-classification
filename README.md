# Image Classification

A comparison of two image classification methods in Python :

- k-nearest neighbors (KNN)
- and artificial neural networks (NN)

The data used to test these methods comes from the CIFAR-10 dataset : https://www.cs.toronto.edu/~kriz/cifar.html

This repository follows the instructions of this tutorial : https://gitlab.ec-lyon.fr/qgalloue/image_classification_instructions

## Principles

### k-Nearest Neighbors Algorithm

The k-nearest neighbors algorithm, also known as KNN or k-NN, is a non-parametric, supervised learning classifier, which uses proximity to make classifications or predictions about the grouping of an individual data point. While it can be used for either regression or classification problems, it is typically used as a classification algorithm, working off the assumption that similar points can be found near one another.

KNN works by finding the distances between a query and all the examples in the training data, selecting the specified number examples (K) closest to the query, then votes for the most frequent label (in the case of classification) or averages the labels (in the case of regression).

<figure>
<img src="https://www.ibm.com/content/dam/connectedassets-adobe-cms/worldwide-content/cdp/cf/ul/g/ef/3a/KNN.component.xl.ts=1639762044031.png/content/adobe-cms/ca/fr/topics/knn/jcr:content/root/table_of_contents/intro/complex_narrative/items/content_group/image" alt="Trulli" style="width:100%">
<figcaption align = "center"><b>Principle of K Neirest Neighbors (KNN)</b></figcaption>
</figure>

_Source : https://www.ibm.com/topics/knn_

For more references, see [K-Nearest Neighbors Algorithm](https://www.ibm.com/topics/knn) or [Machine Learning Basics with the K-Nearest Neighbors Algorithm](https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761)

### Neural Networks

Neural networks, also known as artificial neural networks (ANNs) or simulated neural networks (SNNs), are a subset of machine learning and are at the heart of deep learning algorithms. Their name and structure are inspired by the human brain, mimicking the way that biological neurons signal to one another.

Artificial neural networks (ANNs) are comprised of a node layers, containing an input layer, one or more hidden layers, and an output layer. Each node, or artificial neuron, connects to another and has an associated weight and threshold. If the output of any individual node is above the specified threshold value, that node is activated, sending data to the next layer of the network. Otherwise, no data is passed along to the next layer of the network.

<figure>
<img src="https://1.cms.s81c.com/sites/default/files/2021-01-06/ICLH_Diagram_Batch_01_03-DeepNeuralNetwork-WHITEBG.png" alt="Trulli" style="width:100%">
<figcaption align = "center"><b>Principle of Artificial Neural Network (NN)</b></figcaption>
</figure>

Neural networks rely on training data to learn and improve their accuracy over time. However, once these learning algorithms are fine-tuned for accuracy, they are powerful tools in computer science and artificial intelligence, allowing us to classify and cluster data at a high velocity. Tasks in speech recognition or image recognition can take minutes versus hours when compared to the manual identification by human experts.

_Source : https://www.ibm.com/cloud/learn/neural-networks_

## CIFAR-10

The CIFAR-10 and CIFAR-100 are labeled subsets of the 80 million tiny images dataset. They were collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton.

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

Here are the classes in the dataset, as well as 10 random images from each: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

The classes are completely mutually exclusive. There is no overlap between automobiles and trucks. "Automobile" includes sedans, SUVs, things of that sort. "Truck" includes only big trucks. Neither includes pickup trucks.

To learn more about CIFAR data or download data, click the Source link.

_Source : https://www.cs.toronto.edu/~kriz/cifar.html_

## Installation

### Requirements

This project requires [python3](https://www.python.org/), and common libraries installations :

- [Matplotlib](https://matplotlib.org/) for creating visualizations
- [NumPy](https://numpy.org/) for computing with 2-D arrays
- [SciPy](https://scipy.org/) for fundamental algorithms
- [Pickel](https://docs.python.org/3/library/pickle.html) for Python object serialization
- [Pytest](https://docs.pytest.org/en/7.1.x/) for unit tests

### Development Environment

This project was developped using Visual Studio Code (see https://code.visualstudio.com/ for installation), and contains a vscode [settings.json](.vscode/settings.json) to allow type checking (to check the correct use of functions' outputs.)

It follows the [PEP8](https://peps.python.org/pep-0008/) and [PEP257](https://peps.python.org/pep-0257/) Recommandations.

You can use Python utilites / libraries such as :

- [Black](https://black.readthedocs.io/en/stable/) to format code
- [isort](https://pycqa.github.io/isort/) to sort imports
- [Pydocstyle](http://www.pydocstyle.org/en/stable/) to help you document your code properly

Or you can use VSCode extensions such as :

- [Prettier](https://prettier.io/) to format code
- [autoDocString](https://github.com/NilsJPWerner/autoDocstring) to help you document your code properly
- [Makefile Tools](https://github.com/Microsoft/vscode-makefile-tools/) to support Makefiles in VSCode

## Usage

To run the script contained in main.py simply use :

```Makefile
make run
```

Then, you will be able to choose which method to use by entering 1 or 2 :

- 1 for KNN
- 2 for Neural Network

To run unit tests you may use :

```Makefile
make unittest
```

To visualize test coverage run :

```Makefile
make coverage
```

Unit tests and test coverage reports are run and created with Pytest.

Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Architecture

The [Makefile](Makefile) groups all usefull commands for this project :

- Running the main.py script
- Executing unit tests
- Checking cove coverage

The [main script](main.py) is the executable part of this project. It can be easily run using the method contained in the [Makefile](Makefile).

Functions used for each classification method are contained in files under the directory [modules/](modules/).

Unit tests for each of these functions appear in the [tests/](tests/) directory under the name _test+\_[method_name].py_.

The [vscode settings](.vscode/settings.json) allow type checking for Visual Studio Code Users. It is usefull to check the correct use of functions' outputs during the development phase.

## Support / Contributing

If you want to propose any improvements or need any help, feel free to contribute by [opening an issue](https://gitlab.ec-lyon.fr/colasa/image-classification/-/issues/new).

## Roadmap

Ideas of improvement :

- [ ] OPTIMIZATION : optimize computation for KNN method ;
  - [ ] #1 : OPTIMIZE distance_matrix() : use matrix instead of vectors and return a matrix of 50000 x 10000
  - [x] #2 : OPTIMIZE evaluate_knn() : when computing for k=20, we also compute for all k in [1,2,...,20]. So we could just use the labels retrieved with k=20 and use them for fewer k, instead of re-computing distance_matrix() for each k
- [ ] VISUALISATION : show the computed accuracy points lively ;
- [ ] ARCHITECTURE : group functions in classes (1 class for each classification method, and 1 class to prepare the CIFAR-10 data).

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
