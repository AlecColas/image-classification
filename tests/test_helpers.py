from matplotlib import pyplot as plt

import helpers


def test_choose_classification_method(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "1")
    output = helpers.choose_classification_method()
    assert output == 1


def test_choose_split_factor(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "0.5")
    output = helpers.choose_split_factor()
    assert output == 0.5


def test_choose_to_save(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "yes")
    output = helpers.choose_to_save()
    assert output == 1
