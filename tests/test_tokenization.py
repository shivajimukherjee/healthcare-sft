"""Tests for tokenization and completion-only loss masking.

The prefix-stability test is the important one: our masking assumes that
tokenizing the prompt alone yields exactly the first N tokens of the full
conversation. If a tokenizer ever violated that, labels would silently
misalign by a token or two and training would quietly degrade.
"""

import torch
from transformers import AutoTokenizer

from src.prompts import build_classification_messages
from src.tokenization import (
    IGNORE_INDEX,
    PadCollator,
    build_training_example,
    render_prompt,
)

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LABELS = ["allergy", "malaria", "psoriasis"]


def _tok():
    return AutoTokenizer.from_pretrained(BASE_MODEL)


def test_masked_prefix_is_the_whole_prompt_not_a_token_or_two():
    tok = _tok()
    msgs = build_classification_messages("itchy rash", LABELS, answer="psoriasis")
    ex = build_training_example(msgs, tok, max_length=1024)

    n_masked = sum(1 for t in ex["labels"] if t == IGNORE_INDEX)
    # Regression guard. transformers 5.x returns a BatchEncoding from
    # apply_chat_template(tokenize=True); len() on it is 2, so a naive
    # implementation masks two tokens and trains on its own prompt without
    # ever raising. The prompt here is dozens of tokens long.
    assert n_masked > 20
    # And the masked region must be a prefix, never scattered.
    assert ex["labels"][:n_masked] == [IGNORE_INDEX] * n_masked


def test_input_ids_and_labels_have_equal_length():
    tok = _tok()
    msgs = build_classification_messages("itchy rash", LABELS, answer="psoriasis")
    ex = build_training_example(msgs, tok, max_length=1024)
    assert len(ex["input_ids"]) == len(ex["labels"])


def test_prompt_tokens_are_masked_and_answer_tokens_are_not():
    tok = _tok()
    msgs = build_classification_messages("itchy rash", LABELS, answer="psoriasis")
    ex = build_training_example(msgs, tok, max_length=1024)

    # Some prefix must be masked (the system + user turns) ...
    assert ex["labels"][0] == IGNORE_INDEX
    # ... and some suffix must be supervised (the assistant turn).
    assert ex["labels"][-1] != IGNORE_INDEX


def test_unmasked_labels_decode_to_the_answer():
    tok = _tok()
    msgs = build_classification_messages("itchy rash", LABELS, answer="psoriasis")
    ex = build_training_example(msgs, tok, max_length=1024)

    supervised = [t for t in ex["labels"] if t != IGNORE_INDEX]
    decoded = tok.decode(supervised)
    # This is the whole point of completion-only loss: gradient flows from the
    # answer, not from the system prompt that is identical in every example.
    assert "psoriasis" in decoded


def test_overlong_example_returns_none():
    tok = _tok()
    msgs = build_classification_messages("itchy rash " * 500, LABELS, answer="psoriasis")
    # Dropping beats truncating: a truncated prompt would teach the model to
    # produce an answer it cannot see the evidence for, i.e. to hallucinate.
    assert build_training_example(msgs, tok, max_length=128) is None


def test_render_prompt_ends_with_assistant_header():
    tok = _tok()
    msgs = build_classification_messages("itchy rash", LABELS)
    assert render_prompt(msgs, tok).rstrip().endswith("<|im_start|>assistant")


def test_collator_pads_inputs_with_pad_id_and_labels_with_ignore_index():
    tok = _tok()
    collator = PadCollator(pad_token_id=tok.pad_token_id)
    batch = collator([
        {"input_ids": [1, 2, 3], "labels": [IGNORE_INDEX, 2, 3]},
        {"input_ids": [4, 5], "labels": [IGNORE_INDEX, 5]},
    ])

    assert batch["input_ids"].shape == (2, 3)
    assert batch["input_ids"][1, 2].item() == tok.pad_token_id
    # Padding must be ignored by the loss, not learned as a target.
    assert batch["labels"][1, 2].item() == IGNORE_INDEX
    # Attention mask must hide padding from the attention computation.
    assert batch["attention_mask"][1].tolist() == [1, 1, 0]
    assert batch["labels"].dtype == torch.long
