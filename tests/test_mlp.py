import numpy as np
import pytest

from modules.mlp import (
    one_hot,
    softmax,
)

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = np.array([7, 8, 9])

a1 = np.array([a, a, a])
b1 = np.array([b, b])
c1 = np.array([c, c])


def test_softmax():
    with pytest.raises(np.AxisError):
        softmax(0)
    with pytest.raises(np.AxisError):
        softmax(a)

    res = [np.exp(1 - 3), np.exp(2 - 3), np.exp(3 - 3)]
    res1 = np.array([res, res, res])
    sum = np.exp(1 - 3) + np.exp(2 - 3) + np.exp(3 - 3)
    assert np.array_equal(softmax(a1), res1 / sum)

    return


def test_one_hot():
    hot_a = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    assert np.array_equal(one_hot(a), hot_a)

    with pytest.raises(IndexError):
        one_hot(a1)
    with pytest.raises(AttributeError):
        one_hot(0)
    with pytest.raises(AttributeError):
        one_hot([0, 1])
    return
