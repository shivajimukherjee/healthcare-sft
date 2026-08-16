"""Prompts are the contract between training and evaluation.

These tests exist because a silent drift between the training prompt and the
evaluation prompt would invalidate every number the project reports, while
still 'working'.
"""

import pytest

from src.prompts import (
    CLASSIFY_TAG,
    REQUIRED_SECTIONS,
    SUMMARIZE_TAG,
    build_classification_messages,
    build_summarization_messages,
    normalize_label,
)

LABELS = ["allergy", "malaria", "psoriasis"]


def test_normalize_lowercases_and_strips():
    assert normalize_label("  Drug Reaction  ") == "drug reaction"


def test_normalize_collapses_internal_whitespace():
    assert normalize_label("urinary\n tract  infection") == "urinary tract infection"


def test_normalize_strips_surrounding_punctuation():
    # Base-model outputs routinely arrive as "Malaria." — without this the
    # baseline gets scored down for punctuation rather than for being wrong.
    assert normalize_label("Malaria.") == "malaria"
    assert normalize_label('"malaria"') == "malaria"


def test_classification_inference_messages_have_no_assistant_turn():
    msgs = build_classification_messages("itchy red rash", LABELS)
    assert [m["role"] for m in msgs] == ["system", "user"]


def test_classification_training_messages_end_with_answer():
    msgs = build_classification_messages("itchy red rash", LABELS, answer="psoriasis")
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    # The target is the bare label: no preamble, no punctuation, nothing to parse.
    assert msgs[-1]["content"] == "psoriasis"


def test_classification_system_prompt_lists_every_label():
    msgs = build_classification_messages("itchy red rash", LABELS)
    system = msgs[0]["content"]
    for label in LABELS:
        assert label in system


def test_classification_user_turn_carries_task_tag_and_text():
    msgs = build_classification_messages("itchy red rash", LABELS)
    assert CLASSIFY_TAG in msgs[1]["content"]
    assert "itchy red rash" in msgs[1]["content"]


def test_summarization_system_prompt_names_all_required_sections():
    msgs = build_summarization_messages("Doctor: hello")
    system = msgs[0]["content"]
    for section in REQUIRED_SECTIONS:
        assert section in system


def test_summarization_user_turn_carries_task_tag_and_dialogue():
    msgs = build_summarization_messages("Doctor: hello")
    assert SUMMARIZE_TAG in msgs[1]["content"]
    assert "Doctor: hello" in msgs[1]["content"]


def test_task_tags_are_distinct():
    # The tag is the model's cheapest possible signal for which job to do.
    assert CLASSIFY_TAG != SUMMARIZE_TAG


@pytest.mark.parametrize(
    "builder,args",
    [
        (build_classification_messages, ("text", LABELS)),
        (build_summarization_messages, ("text",)),
    ],
)
def test_no_disclaimer_leaks_into_prompts(builder, args):
    # Disclaimers belong in infer.py and the README. A disclaimer inside a
    # training target would be learned as output and would wreck ROUGE.
    joined = " ".join(m["content"] for m in builder(*args)).lower()
    assert "not medical advice" not in joined
