"""Tests for label-vocabulary derivation.

These run offline on fixtures. The dataset download is exercised by actually
running main(), not by a unit test — network calls in the test suite make it
slow and flaky.
"""

from src.data_prep import derive_labels


def test_derive_labels_deduplicates_and_sorts():
    assert derive_labels(["Malaria", "allergy", "malaria "]) == ["allergy", "malaria"]


def test_derive_labels_returns_stable_order_regardless_of_input_order():
    # The label list is injected into every prompt; if its order changed between
    # runs, the training and evaluation prompts would silently differ.
    assert derive_labels(["b", "a", "c"]) == derive_labels(["c", "a", "b"])
