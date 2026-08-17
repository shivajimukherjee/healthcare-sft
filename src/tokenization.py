"""Turn chat messages into training tensors, with completion-only loss masking.

WHAT SFT ACTUALLY IS
--------------------
Supervised fine-tuning is ordinary next-token prediction on (prompt, answer)
pairs. The only thing that makes it "supervised fine-tuning" rather than plain
language modelling is WHERE the loss is computed: on the answer tokens only.

HOW MASKING WORKS
-----------------
PyTorch's cross-entropy loss ignores any target equal to -100. So we build a
`labels` list that is a copy of `input_ids`, then overwrite every position
belonging to the system and user turns with -100. The model still SEES those
tokens (they stay in input_ids, so attention can use them); it just is not
graded on predicting them.

Without this, most of the gradient signal would come from reproducing a system
prompt that is byte-identical in all ~2000 examples — capacity spent learning
nothing.

Libraries such as TRL automate this. We do it by hand because it is thirty
lines, because it removes a fast-moving dependency, and because it is worth
being able to explain.
"""

from collections.abc import Mapping

import torch

IGNORE_INDEX = -100  # the value torch's cross-entropy skips


def _token_ids(encoded):
    """Normalize apply_chat_template(tokenize=True) output to a flat list[int].

    This is version-defensive on purpose. transformers 4.x returns a plain
    list[int] here; transformers 5.x returns a BatchEncoding (dict-like).
    Calling len() on a BatchEncoding returns 2 — the number of keys — so a
    naive implementation would mask exactly two tokens and silently train the
    model on its own prompt. That bug does not raise, so we normalize once,
    here, and let every caller assume a flat list.

    Note the Mapping check rather than `isinstance(encoded, dict)`:
    BatchEncoding subclasses collections.UserDict, which is NOT a dict
    subclass, so a dict check silently misses and we fall through to
    `list(encoded)` — which yields the mapping's KEYS. Same two-token bug by a
    different route.
    """
    if isinstance(encoded, Mapping):  # BatchEncoding is a UserDict, not a dict
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):  # torch/np tensor
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):  # a batch of one
        encoded = encoded[0]
    return list(encoded)


def render_prompt(messages, tokenizer):
    """Render messages as a string ending in the assistant header.

    `add_generation_prompt=True` appends `<|im_start|>assistant\\n`, which tells
    the model it is now its turn. Used at inference time only.
    """
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def build_training_example(messages, tokenizer, max_length):
    """Tokenize one conversation and mask everything before the assistant turn.

    Returns None when the example is longer than max_length. We drop rather than
    truncate: truncating a dialogue while keeping its summary would train the
    model to invent details it was never shown.
    """
    # Tokenize the prompt alone (system + user), with the assistant header
    # appended. Its length is exactly how many positions we must mask.
    prompt_ids = _token_ids(
        tokenizer.apply_chat_template(
            messages[:-1], tokenize=True, add_generation_prompt=True
        )
    )
    # Tokenize the whole conversation including the assistant answer.
    full_ids = _token_ids(
        tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        )
    )

    # Our masking assumes prompt_ids is a literal prefix of full_ids. That holds
    # for ChatML-style templates because the special tokens are atomic, but a
    # silent violation would misalign every label, so we check instead of hope.
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "Chat template is not prefix-stable; loss masking would misalign."
        )

    if len(full_ids) > max_length:
        return None

    labels = [IGNORE_INDEX] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    return {"input_ids": full_ids, "labels": labels}


class PadCollator:
    """Pad a batch to its longest example.

    Two different pad values are needed, and confusing them is a classic bug:
      - input_ids  -> pad_token_id, and attention_mask 0 so attention ignores it
      - labels     -> IGNORE_INDEX, so the loss ignores it
    Padding labels with pad_token_id instead would train the model to emit
    padding.
    """

    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)

        input_ids, attention_mask, labels = [], [], []
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append([1] * len(feature["input_ids"]) + [0] * pad_len)
            labels.append(feature["labels"] + [IGNORE_INDEX] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
