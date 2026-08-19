"""Tests for the metrics that produce every number in the README.

extract_label is deliberately generous to messy output. That generosity helps
the BASELINE far more than the tuned model, which is the point: an unflattering
baseline would make the reported improvement meaningless.
"""

import pytest

from src.metrics import (
    INVALID,
    classification_metrics,
    extract_label,
    rouge_metrics,
    section_adherence,
)

LABELS = ["allergy", "malaria", "psoriasis"]


def test_extract_label_accepts_exact_match():
    assert extract_label("malaria", LABELS) == "malaria"


def test_extract_label_ignores_case_and_punctuation():
    assert extract_label("Malaria.", LABELS) == "malaria"


def test_extract_label_finds_a_label_inside_a_sentence():
    # Typical untuned-model output. Crediting it is charity toward the baseline.
    assert extract_label("I believe this is malaria, based on the fever.", LABELS) == "malaria"


def test_extract_label_returns_invalid_when_nothing_matches():
    assert extract_label("I am not able to diagnose this.", LABELS) == INVALID


def test_extract_label_prefers_the_longest_match():
    labels = ["infection", "urinary tract infection"]
    assert extract_label("likely urinary tract infection", labels) == "urinary tract infection"


def test_classification_metrics_all_correct():
    m = classification_metrics(["malaria", "allergy"], ["malaria", "allergy"], LABELS)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == pytest.approx(1.0)
    assert m["invalid_rate"] == 0.0
    assert m["n"] == 2


def test_classification_metrics_counts_invalid_as_wrong():
    m = classification_metrics(["I cannot say", "allergy"], ["malaria", "allergy"], LABELS)
    assert m["accuracy"] == 0.5
    assert m["invalid_rate"] == 0.5


def test_classification_metrics_macro_f1_penalises_ignoring_a_class():
    # Predicting one class for everything scores well on accuracy when that
    # class is common. Macro-F1 averages per class, so it does not.
    # Exactly: F1(malaria) = 0.889, F1(allergy) = 0.0 -> macro = 0.444, against
    # an accuracy of 0.800. That gap is the whole reason macro-F1 is reported.
    golds = ["malaria"] * 8 + ["allergy"] * 2
    m = classification_metrics(["malaria"] * 10, golds, LABELS)
    assert m["accuracy"] == pytest.approx(0.8)
    assert m["macro_f1"] == pytest.approx(0.444, abs=0.001)


def test_macro_f1_ignores_classes_with_no_gold_examples():
    # "psoriasis" is in LABELS but absent from these golds. Its F1 is undefined,
    # not zero, so it must not be averaged in — otherwise a perfect model would
    # score 2/3 simply because the test split happened to omit a rare condition.
    m = classification_metrics(["malaria", "allergy"], ["malaria", "allergy"], LABELS)
    assert m["macro_f1"] == pytest.approx(1.0)


def test_section_adherence_requires_every_section():
    good = "Symptoms: cough\nDiagnosis: flu\nHistory of Patient: none\nPlan of Action: rest"
    bad = "Symptoms: cough\nDiagnosis: flu"
    assert section_adherence([good]) == 1.0
    assert section_adherence([bad]) == 0.0
    assert section_adherence([good, bad]) == 0.5


def test_section_adherence_is_case_insensitive():
    text = "SYMPTOMS: cough\nDIAGNOSIS: flu\nHISTORY OF PATIENT: none\nPLAN OF ACTION: rest"
    assert section_adherence([text]) == 1.0


def test_rouge_is_perfect_for_identical_text():
    m = rouge_metrics(["the patient has a fever"], ["the patient has a fever"])
    assert m["rouge1"] == pytest.approx(1.0)
    assert m["rougeL"] == pytest.approx(1.0)


def test_rouge_is_low_for_unrelated_text():
    m = rouge_metrics(["completely unrelated words here"], ["the patient has a fever"])
    assert m["rouge1"] < 0.3
