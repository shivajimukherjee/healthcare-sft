# Healthcare LoRA SFT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a single LoRA adapter on `Qwen2.5-1.5B-Instruct` that both classifies symptom descriptions into 22 conditions and summarizes doctor–patient dialogues into structured clinical notes, evaluated honestly against the untuned base model.

**Architecture:** Pure, unit-tested Python modules (`prompts`, `tokenization`, `metrics`) form a testable core that runs locally on CPU. Two thin scripts (`train.py`, `evaluate.py`) compose that core with GPU work and run on a free Colab T4. Prompt construction lives in exactly one module so the baseline and the tuned model are guaranteed to see identical inputs — this is what makes the comparison trustworthy.

**Tech Stack:** Python 3.11, `transformers`, `peft`, `bitsandbytes`, `datasets`, `scikit-learn`, `rouge-score`, `matplotlib`, `pytest`. Training on Google Colab (T4, 16 GB); everything else local (Apple M1, 8 GB).

**Spec:** [docs/superpowers/specs/2026-08-17-healthcare-lora-sft-design.md](../specs/2026-08-17-healthcare-lora-sft-design.md)

## Global Constraints

- **Base model:** `Qwen/Qwen2.5-1.5B-Instruct` — exact string, used everywhere.
- **Local Python:** 3.11 via `uv venv --python 3.11`. The system Python is 3.14, which many ML wheels do not yet support. Never install into system Python.
- **T4 hardware limits:** fp16 only (**no bf16** — T4 is Turing), `attn_implementation="sdpa"` (**no flash-attention-2** — requires Ampere+).
- **No TRL.** Tokenization and loss masking are hand-written against `transformers.Trainer`. Rationale: TRL's `SFTTrainer` API changes frequently between releases and is the highest-risk dependency in this stack; hand-writing ~20 lines removes that risk and makes the mechanism explainable.
- **Prompts are defined once,** in `src/prompts.py`. `data_prep.py` and `evaluate.py` both import from it. Never inline a prompt string anywhere else.
- **Seed 42** everywhere shuffling or sampling occurs.
- **Test splits are never used for training** and the label list is derived from the train split only.
- **`max_seq_length = 1024`**; examples exceeding it are dropped, never truncated.
- **Explanation is a deliverable.** Every module opens with a docstring covering what and why; every non-obvious ML decision gets an inline comment at its site. Code that works but cannot be explained is a failed deliverable.
- **No safety disclaimer in training targets.** Disclaimers belong in `infer.py` output and the README only.

---

### Task 1: Project scaffold and environment

**Files:**
- Create: `.gitignore`
- Create: `requirements.txt`
- Create: `requirements-train.txt`
- Create: `src/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/test_environment.py`

**Interfaces:**
- Consumes: nothing
- Produces: a working `.venv` at the repo root; `pytest` runnable from the repo root with `src` importable as a package.

- [ ] **Step 1: Create `.gitignore`**

```gitignore
.venv/
__pycache__/
*.pyc
.DS_Store
.pytest_cache/
checkpoints/
*.ipynb_checkpoints
```

- [ ] **Step 2: Create `requirements.txt` (local — CPU only, no training)**

```
# Local environment: data prep, metrics, tests. No GPU training here.
# Version floors, not exact pins: these APIs are stable and floors avoid
# resolver deadlock on macOS ARM. Exact resolved versions are captured in
# requirements.lock.txt in Step 6.
torch>=2.2
transformers>=4.44
datasets>=2.20
scikit-learn>=1.4
rouge-score>=0.1.2
matplotlib>=3.8
pytest>=8.0
```

- [ ] **Step 3: Create `requirements-train.txt` (Colab — GPU training only)**

```
# Colab T4 environment. torch is preinstalled by Colab and deliberately
# NOT pinned here — pinning it forces a multi-GB reinstall and often
# breaks the CUDA build Colab ships.
transformers>=4.44
datasets>=2.20
peft>=0.12
bitsandbytes>=0.43
accelerate>=0.33
```

- [ ] **Step 4: Create empty package markers**

Create `src/__init__.py` and `tests/__init__.py`, both empty files.

- [ ] **Step 5: Create the environment and install**

```bash
cd /Users/admin/Developer/healthcare-sft
uv venv --python 3.11
uv pip install -r requirements.txt
```

Expected: a `.venv/` directory and a successful install with no resolver errors.

- [ ] **Step 6: Capture exact resolved versions**

```bash
.venv/bin/python -m pip freeze | grep -iE '^(torch|transformers|datasets|scikit-learn|rouge-score|matplotlib|pytest)=' > requirements.lock.txt
cat requirements.lock.txt
```

Expected: seven lines with exact `==` versions. This is the reproducibility record the README will reference.

- [ ] **Step 7: Write the environment test**

Create `tests/test_environment.py`:

```python
"""Smoke test: the environment can load the tokenizer we build every prompt with.

If this fails, nothing downstream can work, so it is worth its own test.
"""

from transformers import AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def test_tokenizer_loads_and_has_chat_template():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    # Every prompt in this project is rendered through the chat template.
    # A base (non-Instruct) model would have no template and silently
    # produce garbage prompts, so assert it exists.
    assert tok.chat_template is not None


def test_chat_template_produces_chatml_markers():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    text = tok.apply_chat_template(
        [{"role": "user", "content": "hello"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    # Qwen uses ChatML. The generation prompt must end with the assistant
    # header so the model knows it is its turn to speak.
    assert "<|im_start|>user" in text
    assert text.rstrip().endswith("<|im_start|>assistant")
```

- [ ] **Step 8: Run the test**

Run: `.venv/bin/python -m pytest tests/test_environment.py -v`
Expected: 2 passed. First run downloads the tokenizer (~10 MB), so allow a few seconds.

- [ ] **Step 9: Commit**

```bash
git add .gitignore requirements.txt requirements-train.txt requirements.lock.txt src/__init__.py tests/__init__.py tests/test_environment.py
git commit -m "chore: scaffold project and pin environment

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Prompt construction (`src/prompts.py`)

The single source of truth for what the model sees. Isolating it here is what
guarantees the base-model baseline and the tuned model receive byte-identical
prompts — the property the whole evaluation rests on.

**Files:**
- Create: `src/prompts.py`
- Test: `tests/test_prompts.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `CLASSIFY_TAG: str`, `SUMMARIZE_TAG: str`
  - `REQUIRED_SECTIONS: list[str]`
  - `normalize_label(text: str) -> str`
  - `build_classification_messages(symptom_text: str, labels: list[str], answer: str | None = None) -> list[dict]`
  - `build_summarization_messages(dialogue: str, answer: str | None = None) -> list[dict]`

  When `answer` is `None` the returned list ends with the user turn (inference).
  When `answer` is given, an assistant turn is appended (training).

  `normalize_label` lives here rather than in `data_prep` so that `metrics.py`
  can use it without importing `datasets` — keeping the test suite fast and the
  dependency graph pointing one way.

- [ ] **Step 1: Write the failing test**

Create `tests/test_prompts.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_prompts.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.prompts'`

- [ ] **Step 3: Write the implementation**

Create `src/prompts.py`:

```python
"""Prompt construction for both tasks.

WHY THIS IS ITS OWN MODULE
--------------------------
The headline result of this project is "tuned model vs. base model on the same
test set". That comparison is only meaningful if both models receive exactly
the same prompt. Defining prompts in one place, imported by both data_prep.py
and evaluate.py, makes divergence impossible rather than merely unlikely.

WHY THE LABEL LIST IS IN THE PROMPT
-----------------------------------
We could omit the 22 labels and let the model memorise them in the adapter
weights. We include them instead, for two reasons:

1. Fairness. The baseline is an untuned model; asked to pick from an unstated
   label set it would fail for the wrong reason, and the comparison would be
   rigged in our favour.
2. It sharpens the claim. With the labels visible to both models, the tuned
   model is not being credited with memorising a label space — it is being
   credited with reliably mapping symptom language onto it, and with obeying
   the output format. That is the skill SFT actually teaches.
"""

CLASSIFY_TAG = "[TASK: CLASSIFY]"
SUMMARIZE_TAG = "[TASK: SUMMARIZE]"


def normalize_label(text):
    """Canonical form for a label: lowercase, single-spaced, unpunctuated.

    Applied to gold labels in data_prep and to model predictions in metrics, so
    the baseline is not penalised for cosmetic differences like a trailing full
    stop. Being charitable to the baseline is what makes the reported gain
    credible.

    It lives in this module because the label vocabulary is part of the prompt
    contract, and because metrics.py needs it without pulling in `datasets`.
    """
    cleaned = " ".join(text.strip().lower().split())
    return cleaned.strip('.,;:!?"\'')

# The four headers the MTS-Dialog reference notes use. Also the basis of the
# section-adherence metric in metrics.py — keep the two in sync.
REQUIRED_SECTIONS = ["Symptoms", "Diagnosis", "History of Patient", "Plan of Action"]

_CLASSIFY_SYSTEM = """You are a clinical triage assistant. Given a patient's \
description of their symptoms, identify the single most likely condition.

Respond with only the condition name, exactly as written in the list below. \
Do not explain your reasoning.

Valid conditions:
{labels}"""

_SUMMARIZE_SYSTEM = """You are a clinical documentation assistant. Summarize \
the following doctor-patient conversation into a structured clinical note.

Use exactly these four sections, each on its own line:
{sections}

Write "N/A" for any section the conversation does not cover."""


def build_classification_messages(symptom_text, labels, answer=None):
    """Build the chat messages for one classification example.

    Passing `answer` produces a training example (with the assistant turn);
    omitting it produces an inference prompt. Using one function for both is
    what keeps train-time and test-time prompts identical.
    """
    system = _CLASSIFY_SYSTEM.format(labels="\n".join(f"- {label}" for label in labels))
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"{CLASSIFY_TAG}\n{symptom_text}"},
    ]
    if answer is not None:
        # The target is the bare label. Any preamble ("The condition is...")
        # would have to be stripped at eval time, adding a parsing failure mode.
        messages.append({"role": "assistant", "content": answer})
    return messages


def build_summarization_messages(dialogue, answer=None):
    """Build the chat messages for one summarization example.

    See build_classification_messages for why one function serves both
    training and inference.
    """
    system = _SUMMARIZE_SYSTEM.format(
        sections="\n".join(f"{section}:" for section in REQUIRED_SECTIONS)
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"{SUMMARIZE_TAG}\n{dialogue}"},
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    return messages
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_prompts.py -v`
Expected: 12 passed.

- [ ] **Step 5: Commit**

```bash
git add src/prompts.py tests/test_prompts.py
git commit -m "feat: add prompt construction for both tasks

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: Tokenization and loss masking (`src/tokenization.py`)

The mechanical heart of SFT. This is the part most tutorials hide inside a
library call; here it is 30 explicit lines.

**Files:**
- Create: `src/tokenization.py`
- Test: `tests/test_tokenization.py`

**Interfaces:**
- Consumes: nothing from earlier tasks
- Produces:
  - `IGNORE_INDEX: int` (= -100)
  - `build_training_example(messages: list[dict], tokenizer, max_length: int) -> dict | None`
    Returns `{"input_ids": list[int], "labels": list[int]}`, or `None` if the
    example exceeds `max_length`.
  - `render_prompt(messages: list[dict], tokenizer) -> str` — inference-side
    rendering with the generation prompt appended.
  - `PadCollator(pad_token_id: int)` — callable collator producing batched
    `input_ids`, `attention_mask`, `labels` tensors.

- [ ] **Step 1: Write the failing test**

Create `tests/test_tokenization.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_tokenization.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.tokenization'`

- [ ] **Step 3: Write the implementation**

Create `src/tokenization.py`:

```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_tokenization.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/tokenization.py tests/test_tokenization.py
git commit -m "feat: add tokenization with completion-only loss masking

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Data preparation (`src/data_prep.py`)

**Files:**
- Create: `src/data_prep.py`
- Test: `tests/test_data_prep.py`
- Produces: `data/labels.json`, `data/train.jsonl`, `data/test_classification.jsonl`, `data/test_summarization.jsonl`

**Interfaces:**
- Consumes: `src.prompts.build_classification_messages`, `src.prompts.build_summarization_messages`, `src.prompts.normalize_label`, `src.tokenization.build_training_example`
- Produces:
  - `CLASSIFICATION_DATASET: str`, `SUMMARIZATION_DATASET: str`
  - `derive_labels(rows: list[str]) -> list[str]` — sorted, normalized, deduplicated
  - `main()` — CLI entry point writing all four files

  JSONL record schema: `{"task": "classification"|"summarization", "messages": [...]}`

- [ ] **Step 1: Verify the real dataset split names before writing code**

```bash
.venv/bin/python -c "
from datasets import load_dataset
print('CLF', load_dataset('gretelai/symptom_to_diagnosis'))
print('SUM', load_dataset('har1/MTS_Dialogue-Clinical_Note'))
"
```

VERIFIED 2026-08-18. Classification: `train` (853) / `test` (212), columns `input_text` / `output_text` — as assumed. Summarization: **`train` (1301) ONLY — there is no `validation` split**, columns `ID` / `section_header` / `section_text` / `dialogue`. The implementation therefore carves its own 100-row test set with `train_test_split(test_size=100, seed=42)`.

Also verified: `section_text` already uses the four `REQUIRED_SECTIONS` headers and the "N/A" convention, but only 85.7% of gold notes contain all four. That is the honest ceiling for the section-adherence metric and the README must say so.

- [ ] **Step 2: Write the failing test**

Create `tests/test_data_prep.py`:

```python
"""Tests for label-vocabulary derivation.

These run offline on fixtures. The dataset download is exercised by actually
running main() in Step 6, not by a unit test — network calls in the test suite
make it slow and flaky.
"""

from src.data_prep import derive_labels


def test_derive_labels_deduplicates_and_sorts():
    assert derive_labels(["Malaria", "allergy", "malaria "]) == ["allergy", "malaria"]


def test_derive_labels_returns_stable_order_regardless_of_input_order():
    # The label list is injected into every prompt; if its order changed between
    # runs, the training and evaluation prompts would silently differ.
    assert derive_labels(["b", "a", "c"]) == derive_labels(["c", "a", "b"])
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_data_prep.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.data_prep'`

- [ ] **Step 4: Write the implementation**

Create `src/data_prep.py`:

```python
"""Build the mixed multi-task training set and the two held-out test sets.

WHY ONE MIXED TRAINING FILE
---------------------------
A LoRA adapter is a weight diff applied to the whole model, not a task-bound
object. Which behaviour fires is decided at inference time by the prompt. So a
single adapter can serve both tasks, provided the two prompt shapes are clearly
distinguishable — which is what the task tags and the very different output
formats give us. This is also how the base Instruct model was made: multi-task
SFT is the normal case, not a trick.

SPLIT DISCIPLINE
----------------
The classification dataset ships with an author-provided test split; we use it
untouched so the evaluation is not self-graded. The summarization dataset's
100-row `validation` split becomes our test set. Labels are derived from the
TRAINING split only — deriving them from test would leak test information into
every prompt.

LENGTH POLICY
-------------
Examples longer than --max-seq-length are dropped and counted, never truncated.
"""

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from src.prompts import (
    build_classification_messages,
    build_summarization_messages,
    normalize_label,
)
from src.tokenization import build_training_example

CLASSIFICATION_DATASET = "gretelai/symptom_to_diagnosis"
SUMMARIZATION_DATASET = "har1/MTS_Dialogue-Clinical_Note"
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SEED = 42


def derive_labels(rows):
    """Sorted, deduplicated, normalized label vocabulary."""
    return sorted({normalize_label(row) for row in rows})


def _write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return len(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    clf = load_dataset(CLASSIFICATION_DATASET)
    summ = load_dataset(SUMMARIZATION_DATASET)

    # Labels come from train only. See SPLIT DISCIPLINE above.
    labels = derive_labels(clf["train"]["output_text"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "labels.json").write_text(json.dumps(labels, indent=2))
    print(f"Derived {len(labels)} labels from the training split.")

    train_records, dropped = [], {"classification": 0, "summarization": 0}

    for row in clf["train"]:
        messages = build_classification_messages(
            row["input_text"], labels, answer=normalize_label(row["output_text"])
        )
        # Build the tokenized form purely to test whether it fits; train.py
        # rebuilds it. Duplicated work, but it keeps the length policy in one
        # place and the data files human-readable.
        if build_training_example(messages, tokenizer, args.max_seq_length) is None:
            dropped["classification"] += 1
            continue
        train_records.append({"task": "classification", "messages": messages})

    for row in summ["train"]:
        messages = build_summarization_messages(
            row["dialogue"], answer=row["section_text"]
        )
        if build_training_example(messages, tokenizer, args.max_seq_length) is None:
            dropped["summarization"] += 1
            continue
        train_records.append({"task": "summarization", "messages": messages})

    # Shuffle so batches mix both tasks. Training on all of one task then all of
    # the other invites the model to overwrite the first with the second.
    random.Random(SEED).shuffle(train_records)

    # Test sets keep the assistant turn as the reference; evaluate.py strips it
    # before generation and compares against it afterwards.
    clf_test = [
        {
            "task": "classification",
            "messages": build_classification_messages(
                row["input_text"], labels, answer=normalize_label(row["output_text"])
            ),
        }
        for row in clf["test"]
    ]
    summ_test = [
        {
            "task": "summarization",
            "messages": build_summarization_messages(
                row["dialogue"], answer=row["section_text"]
            ),
        }
        for row in summ["validation"]
    ]

    n_train = _write_jsonl(out_dir / "train.jsonl", train_records)
    n_clf = _write_jsonl(out_dir / "test_classification.jsonl", clf_test)
    n_summ = _write_jsonl(out_dir / "test_summarization.jsonl", summ_test)

    n_clf_train = sum(r["task"] == "classification" for r in train_records)
    print(
        f"train.jsonl: {n_train} examples "
        f"({n_clf_train} classification / {n_train - n_clf_train} summarization)"
    )
    print(f"dropped as too long: {dropped}")
    print(f"test_classification.jsonl: {n_clf}   test_summarization.jsonl: {n_summ}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_data_prep.py -v`
Expected: 2 passed.

- [ ] **Step 6: Generate the data**

```bash
.venv/bin/python -m src.data_prep
```

ACTUAL: 22 labels; 2040 training examples (851 classification / 1189 summarization); 11 summarization rows dropped as too long; 3 rows dropped as train/test duplicates; 212 and 100 test examples.

- [ ] **Step 7: Read three formatted examples with your own eyes**

```bash
.venv/bin/python -c "
import json
rows = [json.loads(l) for l in open('data/train.jsonl')]
for r in rows[:3]:
    print('=' * 70, r['task'])
    for m in r['messages']:
        print(f\"--- {m['role']} ---\"); print(m['content'][:400])
"
```

Expected: sane system prompts, the task tag present, and assistant turns that are a bare label or a four-section note. **Do not skip this.** Every downstream number depends on this data being right, and malformed prompts here fail silently rather than loudly.

- [ ] **Step 8: Commit**

```bash
git add src/data_prep.py tests/test_data_prep.py data/
git commit -m "feat: build mixed multi-task dataset from public medical corpora

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: Evaluation metrics (`src/metrics.py`)

**Files:**
- Create: `src/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `src.prompts.normalize_label`, `src.prompts.REQUIRED_SECTIONS`
- Produces:
  - `INVALID: str` (= `"__invalid__"`)
  - `extract_label(raw: str, labels: list[str]) -> str`
  - `classification_metrics(preds: list[str], golds: list[str], labels: list[str]) -> dict`
    keys: `accuracy`, `macro_f1`, `invalid_rate`, `n`
  - `section_adherence(texts: list[str]) -> float`
  - `rouge_metrics(preds: list[str], refs: list[str]) -> dict` keys: `rouge1`, `rouge2`, `rougeL`

- [ ] **Step 1: Write the failing test**

Create `tests/test_metrics.py`:

```python
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
    golds = ["malaria"] * 8 + ["allergy"] * 2
    m = classification_metrics(["malaria"] * 10, golds, LABELS)
    assert m["accuracy"] == pytest.approx(0.8)
    assert m["macro_f1"] < 0.4


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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.metrics'`

- [ ] **Step 3: Write the implementation**

Create `src/metrics.py`:

```python
"""Metrics for both tasks.

WHY EACH METRIC IS HERE
-----------------------
accuracy      Headline number, easy to read, but misleading alone when classes
              are unbalanced.
macro_f1      Averages F1 per class rather than per example, so a model that
              ignores rare conditions cannot hide behind a common one. With 22
              medical conditions, ignoring rare ones is exactly the failure we
              care about.
invalid_rate  Fraction of outputs that are not any valid label. An untuned model
              will chat, hedge, and explain; a tuned one answers with a label.
              This is the single clearest demonstration of what SFT teaches,
              because it isolates format-following from medical correctness.
rouge1/2/L    Standard summarization overlap metric. Genuinely weak — it rewards
              word overlap, not faithfulness — so it is reported alongside
              section adherence and hand-inspected examples, never alone.
section_adherence
              Fraction of summaries containing all four required headers. ROUGE
              cannot see structure; this is what shows the format was learned.
"""

from rouge_score import rouge_scorer
from sklearn.metrics import f1_score

from src.prompts import REQUIRED_SECTIONS, normalize_label

INVALID = "__invalid__"


def extract_label(raw, labels):
    """Recover a label from raw model output, or INVALID if there isn't one.

    Deliberately generous: it accepts a label embedded in a sentence, because an
    untuned model answers "I think this is malaria" rather than "malaria". That
    generosity almost exclusively helps the baseline, which is the point — a
    baseline handicapped by output parsing would inflate our reported gain.
    """
    normalized = normalize_label(raw)
    if normalized in labels:
        return normalized

    # Longest match first, so "urinary tract infection" wins over "infection".
    for label in sorted(labels, key=len, reverse=True):
        if label in normalized:
            return label
    return INVALID


def classification_metrics(preds, golds, labels):
    """Accuracy, macro-F1 and invalid rate for the classification task."""
    extracted = [extract_label(p, labels) for p in preds]
    normalized_golds = [normalize_label(g) for g in golds]

    correct = sum(p == g for p, g in zip(extracted, normalized_golds))
    invalid = sum(p == INVALID for p in extracted)

    # Passing labels= excludes INVALID from the averaged classes, so an invalid
    # prediction counts as a miss on the true class rather than inventing a
    # 23rd class that would dilute the macro average.
    macro_f1 = f1_score(
        normalized_golds, extracted, labels=labels, average="macro", zero_division=0
    )

    return {
        "accuracy": correct / len(preds),
        "macro_f1": float(macro_f1),
        "invalid_rate": invalid / len(preds),
        "n": len(preds),
    }


def section_adherence(texts):
    """Fraction of summaries containing all four required section headers."""
    if not texts:
        return 0.0
    hits = sum(
        all(section.lower() in text.lower() for section in REQUIRED_SECTIONS)
        for text in texts
    )
    return hits / len(texts)


def rouge_metrics(preds, refs):
    """Mean ROUGE-1/2/L F-measure over the test set."""
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for pred, ref in zip(preds, refs):
        scores = scorer.score(ref, pred)
        for key in totals:
            totals[key] += scores[key].fmeasure
    return {key: value / len(preds) for key, value in totals.items()}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_metrics.py -v`
Expected: 12 passed.

- [ ] **Step 5: Run the whole suite**

Run: `.venv/bin/python -m pytest -v`
Expected: 36 passed.

- [ ] **Step 6: Commit**

```bash
git add src/metrics.py tests/test_metrics.py
git commit -m "feat: add classification and summarization metrics

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: Training script and Colab notebook

**Files:**
- Create: `src/train.py`
- Create: `notebooks/train_colab.ipynb`
- Produces: `adapters/qwen-healthcare-lora/`

**Interfaces:**
- Consumes: `src.tokenization.build_training_example`, `src.tokenization.PadCollator`
- Produces: an adapter directory containing `adapter_config.json` and `adapter_model.safetensors`

- [ ] **Step 1: Write the training script**

Create `src/train.py`:

```python
"""QLoRA supervised fine-tuning of Qwen2.5-1.5B-Instruct on the mixed dataset.

WHAT QLoRA IS
-------------
Two ideas stacked:

  Quantization  The base model's frozen weights are stored in 4 bits instead of
                16. A 1.5B model drops from ~3 GB to ~1 GB. The weights are
                dequantized on the fly for each matmul, so quality loss is small.

  LoRA          Every targeted weight matrix W is frozen, and a trainable
                bypass is added beside it:  h = Wx + (alpha/r) * B(Ax)
                where A is (r x d) and B is (d x r) with r=16 << d. We train
                roughly 1% of the parameters, and the saved artifact is a ~40 MB
                adapter rather than a 3 GB model.

                B is initialized to ZERO, so BA = 0 at step 0 and the model
                starts out behaving EXACTLY like the base model. Training moves
                away from a known-good starting point, never from a random one.

WHY THESE HYPERPARAMETERS
-------------------------
lr=2e-4     ~10x a typical full-fine-tune learning rate. Legitimate here because
            only the small adapter matrices move; the base model cannot be
            damaged by a large step.
epochs=3    At ~2000 examples, more begins memorising rather than generalising.
alpha=2*r   Convention. alpha/r is an output SCALING factor, not extra capacity,
            which is why changing r does not force re-tuning the learning rate.
target_modules
            Attention AND MLP projections. Adapting attention only is the common
            shortcut and measurably underperforms.
fp16        The T4 is a Turing GPU with NO bf16 support. Using bf16 here fails.
sdpa        flash_attention_2 requires Ampere or newer; on a T4 it will not load.
paged_adamw_8bit
            8-bit optimizer states, paged to host RAM on a spike. Optimizer state
            is often the thing that OOMs, not the weights.
"""

import argparse
import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from src.tokenization import PadCollator, build_training_example

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SEED = 42


def build_dataset(path, tokenizer, max_seq_length, limit=None):
    """Load the JSONL produced by data_prep and tokenize with loss masking."""
    records = [json.loads(line) for line in Path(path).read_text().splitlines()]
    if limit:
        records = records[:limit]

    examples = []
    for record in records:
        example = build_training_example(
            record["messages"], tokenizer, max_seq_length
        )
        if example is not None:  # data_prep already filtered; belt and braces
            examples.append(example)
    return examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", default="data/train.jsonl")
    parser.add_argument("--output-dir", default="adapters/qwen-healthcare-lora")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument(
        "--limit", type=int, default=None, help="Use N examples only (smoke test)."
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    dataset = build_dataset(
        args.train_file, tokenizer, args.max_seq_length, args.limit
    )
    print(f"Training on {len(dataset)} examples.")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # NF4 is information-theoretically better
                                    # than plain int4 for normally-distributed
                                    # neural network weights.
        bnb_4bit_use_double_quant=True,  # quantizes the quantization constants
        bnb_4bit_compute_dtype=torch.float16,  # fp16: T4 has no bf16
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quant_config,
        device_map="auto",
        attn_implementation="sdpa",  # flash_attention_2 needs Ampere+
    )
    # Casts layernorms to fp32 and enables input gradients, both of which
    # 4-bit training needs to remain numerically stable.
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",   # attention
            "gate_proj", "up_proj", "down_proj",      # MLP
        ],
    )
    model = get_peft_model(model, lora_config)
    # Prints trainable vs total. Expect roughly 1% trainable — if it prints
    # 100%, LoRA did not attach and you are about to full-fine-tune by accident.
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir="checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        fp16=True,          # NOT bf16 — see module docstring
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        report_to="none",   # no wandb prompt in Colab
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=PadCollator(tokenizer.pad_token_id),
    )
    trainer.train()

    # Saves ONLY the adapter (~40 MB), not the 3 GB base model. The adapter is
    # a diff and is meaningless without the base, which adapter_config.json
    # records by name.
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Generate the Colab notebook**

```bash
.venv/bin/pip install nbformat
.venv/bin/python - <<'PY'
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_markdown_cell(
        "# Healthcare LoRA SFT — training\n\n"
        "Runs on a **free Colab T4**. Set `Runtime > Change runtime type > T4 GPU` "
        "before running anything.\n\n"
        "Pipeline: clone repo -> install -> rebuild data -> smoke test -> full train -> "
        "download adapter."
    ),
    nbf.v4.new_code_cell("!nvidia-smi"),
    nbf.v4.new_markdown_cell(
        "## 1. Clone and install\n\n"
        "`torch` is deliberately absent from `requirements-train.txt` — Colab already "
        "ships a CUDA build, and reinstalling it usually breaks the environment."
    ),
    nbf.v4.new_code_cell(
        "!git clone https://github.com/YOUR_USERNAME/healthcare-sft.git\n"
        "%cd healthcare-sft\n"
        "!pip install -q -r requirements-train.txt"
    ),
    nbf.v4.new_markdown_cell(
        "## 2. Rebuild the dataset\n\n"
        "The repo already contains `data/`, but regenerating proves the pipeline is "
        "reproducible from source rather than from a committed artifact."
    ),
    nbf.v4.new_code_cell("!python -m src.data_prep"),
    nbf.v4.new_markdown_cell(
        "## 3. Smoke test — 10 examples, 1 epoch\n\n"
        "Catches OOM, dtype and masking errors in ~60 seconds instead of 30 minutes in. "
        "Check that `print_trainable_parameters` reports roughly **1%** trainable; "
        "100% means LoRA failed to attach."
    ),
    nbf.v4.new_code_cell(
        "!python -m src.train --limit 10 --epochs 1 --output-dir /tmp/smoke"
    ),
    nbf.v4.new_markdown_cell(
        "## 4. Full training run\n\n"
        "~770 steps, roughly 20–40 minutes. Watch that the loss falls and does not "
        "become `nan` (an fp16 instability symptom).\n\n"
        "If it OOMs: `--batch-size 1`, then `--max-seq-length 768`."
    ),
    nbf.v4.new_code_cell("!python -m src.train"),
    nbf.v4.new_markdown_cell(
        "## 5. Evaluate — base vs tuned\n\n"
        "Both runs use identical prompts from `src/prompts.py`, greedy decoding, and "
        "the same held-out test sets."
    ),
    nbf.v4.new_code_cell("!python -m src.evaluate --no-adapter --tag base"),
    nbf.v4.new_code_cell(
        "!python -m src.evaluate "
        "--adapter adapters/qwen-healthcare-lora --tag tuned"
    ),
    nbf.v4.new_markdown_cell(
        "## 6. Download the adapter and results\n\n"
        "The adapter is ~40 MB: LoRA matrices only, not the 3 GB base model."
    ),
    nbf.v4.new_code_cell(
        "!zip -r artifacts.zip adapters/ results/\n"
        "from google.colab import files\n"
        "files.download('artifacts.zip')"
    ),
]

Path("notebooks").mkdir(exist_ok=True)
nbf.write(nb, "notebooks/train_colab.ipynb")
print("wrote notebooks/train_colab.ipynb")
PY
```

- [ ] **Step 3: Commit before training**

```bash
git add src/train.py notebooks/train_colab.ipynb
git commit -m "feat: add QLoRA training script and Colab notebook

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

- [ ] **Step 4: Push to GitHub so Colab can clone it**

```bash
gh repo create healthcare-sft --public --source=. --remote=origin --push
```

Then update the clone URL in the notebook's cell 4 to the real repository URL, and commit that change.

- [ ] **Step 5: Run the smoke test on Colab**

Open `notebooks/train_colab.ipynb` in Colab with a T4 runtime and run cells 1–6.
Expected: `print_trainable_parameters` reports roughly 1% trainable; ten steps complete without OOM; a finite loss is printed.
**Stop and fix here if anything fails** — a smoke failure at 60 seconds is far cheaper than the same failure 20 minutes into the full run.

- [ ] **Step 6: Run full training**

Run notebook cell 8 (`!python -m src.train`).
Expected: ~770 steps, 20–40 minutes, loss falling from roughly 2–3 toward under 1, never `nan`.

- [ ] **Step 7: Commit the adapter**

Download `artifacts.zip`, unzip `adapters/` into the repo, then:

```bash
git add adapters/
git commit -m "feat: add trained multi-task LoRA adapter

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: Evaluation runner (`src/evaluate.py`)

**Files:**
- Create: `src/evaluate.py`
- Produces: `results/metrics.json`, `results/confusion_matrix_{tag}.png`, `results/qualitative_examples.md`

**Interfaces:**
- Consumes: `src.metrics.*`, `src.tokenization.render_prompt`
- Produces: `results/metrics.json` keyed by tag (`base` / `tuned`), each holding
  `classification` and `summarization` sub-objects. Task 9 reads this file.

- [ ] **Step 1: Write the evaluation script**

Create `src/evaluate.py`:

```python
"""Evaluate the base model and the tuned model on identical held-out data.

WHY THE BASELINE IS NOT OPTIONAL
--------------------------------
"87% accuracy" means nothing on its own. "62% -> 87% on the same 212 held-out
examples, same prompts, same decoding" is a result. This script therefore runs
in two modes over one code path, so the two runs cannot accidentally differ:

    python -m src.evaluate --no-adapter --tag base
    python -m src.evaluate --adapter adapters/qwen-healthcare-lora --tag tuned

Decoding is greedy (do_sample=False) in both runs so results are reproducible
and no sampling luck enters the comparison.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display in Colab
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.metrics import (
    classification_metrics,
    extract_label,
    rouge_metrics,
    section_adherence,
)
from src.prompts import normalize_label
from src.tokenization import render_prompt

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def load_model(adapter_path=None):
    """Load the 4-bit base model, optionally attaching a LoRA adapter.

    Same quantization in both modes: comparing a 4-bit tuned model against a
    16-bit baseline would confound the adapter's effect with precision.
    """
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quant_config,
        device_map="auto",
        attn_implementation="sdpa",
    )
    if adapter_path:
        from peft import PeftModel

        # Attaching the adapter re-adds the B@A bypass beside each frozen
        # weight. Without this line the exact same code evaluates the baseline.
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model


@torch.no_grad()
def generate(model, tokenizer, messages_list, max_new_tokens, batch_size=8):
    """Greedy-decode a batch of prompts, returning only the new tokens."""
    outputs = []
    for start in range(0, len(messages_list), batch_size):
        chunk = messages_list[start : start + batch_size]
        prompts = [render_prompt(m, tokenizer) for m in chunk]

        # Left padding is required for batched generation: with right padding,
        # generation would continue from pad tokens instead of from the prompt.
        tokenizer.padding_side = "left"
        batch = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

        generated = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy: deterministic and comparable
            pad_token_id=tokenizer.pad_token_id,
        )
        # Slice off the prompt so only the answer remains.
        for row in generated[:, batch["input_ids"].shape[1] :]:
            outputs.append(tokenizer.decode(row, skip_special_tokens=True).strip())
        print(f"  generated {min(start + batch_size, len(messages_list))}/{len(messages_list)}")
    return outputs


def _load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--no-adapter", action="store_true")
    parser.add_argument("--tag", required=True, help="'base' or 'tuned'")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    data_dir, results_dir = Path(args.data_dir), Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    labels = json.loads((data_dir / "labels.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = load_model(None if args.no_adapter else args.adapter)

    # --- classification -----------------------------------------------------
    clf_rows = _load_jsonl(data_dir / "test_classification.jsonl")
    # Drop the assistant turn: that is the answer we are trying to predict.
    clf_prompts = [row["messages"][:-1] for row in clf_rows]
    clf_golds = [row["messages"][-1]["content"] for row in clf_rows]

    print(f"[{args.tag}] classification: {len(clf_prompts)} examples")
    # 16 new tokens is ample for a label and stops the base model rambling
    # for a paragraph, which would only waste time.
    clf_preds = generate(model, tokenizer, clf_prompts, max_new_tokens=16)
    clf_metrics = classification_metrics(clf_preds, clf_golds, labels)
    print(f"[{args.tag}] {clf_metrics}")

    # --- summarization ------------------------------------------------------
    summ_rows = _load_jsonl(data_dir / "test_summarization.jsonl")
    summ_prompts = [row["messages"][:-1] for row in summ_rows]
    summ_refs = [row["messages"][-1]["content"] for row in summ_rows]

    print(f"[{args.tag}] summarization: {len(summ_prompts)} examples")
    summ_preds = generate(model, tokenizer, summ_prompts, max_new_tokens=256, batch_size=4)
    summ_metrics = rouge_metrics(summ_preds, summ_refs)
    summ_metrics["section_adherence"] = section_adherence(summ_preds)
    summ_metrics["n"] = len(summ_preds)
    print(f"[{args.tag}] {summ_metrics}")

    # --- persist ------------------------------------------------------------
    metrics_path = results_dir / "metrics.json"
    all_metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    all_metrics[args.tag] = {
        "classification": clf_metrics,
        "summarization": summ_metrics,
    }
    metrics_path.write_text(json.dumps(all_metrics, indent=2))

    # Confusion matrix. With 22 classes it is dense, but it makes systematic
    # confusions visible in a way a single accuracy number cannot.
    extracted = [extract_label(p, labels) for p in clf_preds]
    fig, ax = plt.subplots(figsize=(14, 12))
    ConfusionMatrixDisplay.from_predictions(
        [normalize_label(g) for g in clf_golds],
        extracted,
        labels=labels,
        xticks_rotation="vertical",
        ax=ax,
        colorbar=False,
    )
    ax.set_title(f"Classification confusion matrix — {args.tag}")
    fig.tight_layout()
    fig.savefig(results_dir / f"confusion_matrix_{args.tag}.png", dpi=120)

    # Qualitative examples: ROUGE cannot show that a summary is well-formed
    # but wrong, or badly-formed but right. Reading five is worth a metric.
    with (results_dir / f"qualitative_{args.tag}.md").open("w") as handle:
        handle.write(f"# Qualitative examples — {args.tag}\n\n")
        for i in range(min(5, len(summ_preds))):
            handle.write(f"## Example {i + 1}\n\n")
            handle.write(f"**Model output**\n\n```\n{summ_preds[i]}\n```\n\n")
            handle.write(f"**Reference**\n\n```\n{summ_refs[i]}\n```\n\n---\n\n")

    print(f"[{args.tag}] wrote results to {results_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the baseline evaluation on Colab**

```bash
!python -m src.evaluate --no-adapter --tag base
```

Expected: a substantial `invalid_rate` (the untuned model chats rather than answering) and low section adherence. This run takes ~5–10 minutes.

- [ ] **Step 3: Run the tuned evaluation on Colab**

```bash
!python -m src.evaluate --adapter adapters/qwen-healthcare-lora --tag tuned
```

Expected: higher accuracy and macro-F1, near-zero `invalid_rate`, and much higher section adherence.

- [ ] **Step 4: Check for task interference**

Read `results/qualitative_tuned.md` and confirm the tuned model does not emit summaries when asked to classify, and vice versa. If `invalid_rate` is *worse* after tuning, that is task interference: retrain with `--lora-r 32` and re-evaluate. Record whatever is observed — a measured negative result is worth reporting.

- [ ] **Step 5: Commit results**

```bash
git add src/evaluate.py results/
git commit -m "feat: add evaluation with base-model baseline

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: Inference CLI and concepts documentation

**Files:**
- Create: `src/infer.py`
- Create: `docs/CONCEPTS.md`

**Interfaces:**
- Consumes: `src.prompts.build_classification_messages`, `src.prompts.build_summarization_messages`, `src.tokenization.render_prompt`
- Produces: a CLI usable for a live demo

  Note: this deliberately does **not** reuse `src.evaluate.load_model`. That
  function loads in 4-bit via bitsandbytes, which has no Apple Silicon support;
  `infer.py` loads fp16 instead so the demo runs on the Mac.

- [ ] **Step 1: Write the inference CLI**

Create `src/infer.py`:

```python
"""Single-example inference — the live demo entry point.

Runs on CPU/MPS locally: the base model in fp16 is ~3 GB, which fits an 8 GB
Mac, so no GPU is needed to demo the result.

    python -m src.infer --task classify --text "burning when urinating, ..."
    python -m src.infer --task summarize --text "Doctor: What brings you in?..."

THE DISCLAIMER LIVES HERE, NOT IN THE WEIGHTS
---------------------------------------------
It is deliberately printed by the wrapper rather than trained into the model.
Training it into the assistant turn would make it part of the learned output,
which would corrupt ROUGE against the references and waste adapter capacity on
a constant string. Safety text is an application-layer concern.
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.prompts import build_classification_messages, build_summarization_messages
from src.tokenization import render_prompt

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

DISCLAIMER = (
    "\n---\nDecision-support demonstration only. Not a medical device and not "
    "medical advice. Outputs are unverified and must be reviewed by a qualified "
    "clinician."
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["classify", "summarize"], required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--adapter", default="adapters/qwen-healthcare-lora")
    parser.add_argument("--labels", default="data/labels.json")
    parser.add_argument("--no-adapter", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    # No 4-bit here: bitsandbytes has no Apple Silicon support, and at 1.5B
    # fp16 fits in 8 GB comfortably anyway.
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, device_map="auto"
    )
    if not args.no_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    if args.task == "classify":
        labels = json.loads(Path(args.labels).read_text())
        messages = build_classification_messages(args.text, labels)
        max_new_tokens = 16
    else:
        messages = build_summarization_messages(args.text)
        max_new_tokens = 256

    inputs = tokenizer(render_prompt(messages, tokenizer), return_tensors="pt").to(
        model.device
    )
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    answer = tokenizer.decode(
        generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()
    print(answer)
    print(DISCLAIMER)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test the CLI locally on both tasks**

```bash
.venv/bin/python -m src.infer --task classify \
  --text "I have a burning feeling when I urinate and I need to go constantly."
.venv/bin/python -m src.infer --task summarize \
  --text "Doctor: What brings you in today? Patient: My right knee has been aching for three weeks, worse on stairs. Doctor: Any injury? Patient: No, it just started."
```

Expected: a bare label from the first, a four-section note from the second, both followed by the disclaimer.

- [ ] **Step 3: Write `docs/CONCEPTS.md`**

Create the standalone explainer with these sections, written in full prose (this is the pre-interview revision document — no bullet fragments):

1. **Where SFT sits** — pretraining vs SFT vs preference tuning; what each stage buys; that SFT is next-token prediction with masked prompts.
2. **What SFT can and cannot teach** — format and task behavior, yes; new medical knowledge, no. Why this project's claims are worded accordingly.
3. **LoRA** — the `h = Wx + (alpha/r)·B(Ax)` decomposition; why `B` starts at zero; what `r` and `alpha` each control; why attention *and* MLP are targeted.
4. **What an adapter file is** — `adapter_config.json` + `adapter_model.safetensors`, ~40 MB, a diff not a model; hot-swapping one base with many adapters; `merge_and_unload` and the latency/flexibility tradeoff.
5. **QLoRA** — 4-bit NF4, double quantization, why quantizing a *frozen* base is safe.
6. **Completion-only loss masking** — the `-100` mechanism, why prompt tokens are excluded, pointing at `src/tokenization.py` as the implementation. Note that TRL's `SFTTrainer` automates this and why it was done by hand here.
7. **Multi-task in one adapter** — prompt-conditioned behavior; task interference and how §7 of the spec detects it.
8. **The metrics** — what each measures, why macro-F1 over accuracy, why invalid-rate is the clearest SFT signal, and why ROUGE is weak.
9. **Hardware constraints** — T4 has no bf16 and no flash-attention-2; why fp16 and sdpa; why bitsandbytes is unavailable on Apple Silicon.
10. **What I would do next** — separate per-client adapters, constrained decoding for classification, LLM-as-judge for summaries, a larger label space, calibration and abstention.

- [ ] **Step 4: Commit**

```bash
git add src/infer.py docs/CONCEPTS.md
git commit -m "feat: add inference CLI and concepts documentation

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 9: README with results

**Files:**
- Create: `README.md`

**Interfaces:**
- Consumes: `results/metrics.json` (written by Task 7)
- Produces: the repository's front door

- [ ] **Step 1: Read the real numbers**

```bash
cat results/metrics.json
```

Every number in the README comes from this file. **Do not write a number you have not read here.**

- [ ] **Step 2: Write `README.md`**

Structure, with real values substituted into the tables:

````markdown
# Healthcare LoRA SFT

A single LoRA adapter on `Qwen2.5-1.5B-Instruct` that performs two clinical NLP
tasks: symptom-to-condition classification and doctor–patient dialogue
summarization.

> Decision-support demonstration only. Not a medical device, not medical advice.

## Results

All numbers are on held-out test sets, identical prompts for both models,
greedy decoding.

### Classification — 212 held-out examples, 22 conditions

| Metric | Base | Tuned | Δ |
|---|---|---|---|
| Accuracy | … | … | … |
| Macro-F1 | … | … | … |
| Invalid-label rate | … | … | … |

### Summarization — 100 held-out dialogues

| Metric | Base | Tuned | Δ |
|---|---|---|---|
| ROUGE-1 | … | … | … |
| ROUGE-2 | … | … | … |
| ROUGE-L | … | … | … |
| Section adherence | … | … | … |

Confusion matrices: `results/`. Side-by-side summaries:
`results/qualitative_tuned.md`.

## Approach

Base model, why it was chosen, QLoRA configuration table, the single-adapter
multi-task decision, and why generative classification was chosen over a
classification head.

## Data

Both datasets with links, licenses, sizes, split discipline, the length filter
and how many examples it dropped.

## Reproducing

Local setup, `python -m src.data_prep`, the Colab notebook link, both evaluate
commands.

## Repository layout

The tree from the spec, one line each.

## Limitations

- Not a medical device; no PHI handled — both datasets are public and de-identified.
- SFT taught output format and task behavior, not medical knowledge. A 1.5B
  model's clinical knowledge comes from pretraining and is not reliable.
- 22 conditions is a closed toy label space; real triage is open-set.
- ROUGE measures word overlap, not faithfulness. A fluent, well-structured,
  factually wrong summary scores well. Hallucinated findings in a clinical note
  are a patient-safety issue, not a UX issue.
- Any real deployment needs abstention and human-in-the-loop review.

## Why a small local model

Clinical data often cannot leave the building. One on-prem 1.5B base model
with a folder of ~40 MB swappable adapters — one per client, specialty, or note
format — is a serious architecture, not a compromise.

## What I would do next

From `docs/CONCEPTS.md` §10.
````

- [ ] **Step 3: Verify the full test suite still passes**

```bash
.venv/bin/python -m pytest -v
```
Expected: 36 passed.

- [ ] **Step 4: Verify every README number against `results/metrics.json`**

Read them side by side. A wrong number in the README is worse than no README.

- [ ] **Step 5: Commit and push**

```bash
git add README.md
git commit -m "docs: add README with evaluation results

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
git push
```

---

## Definition of Done

- [ ] `.venv/bin/python -m pytest` passes (36 tests)
- [ ] `data/` contains four files; the label count is 22
- [ ] `adapters/qwen-healthcare-lora/` contains `adapter_config.json` and `adapter_model.safetensors`
- [ ] `results/metrics.json` contains **both** `base` and `tuned` entries for both tasks
- [ ] `python -m src.infer` works locally on the Mac for both tasks
- [ ] README numbers match `results/metrics.json` exactly
- [ ] `docs/CONCEPTS.md` covers all ten sections
- [ ] Every file can be explained aloud without reading ahead

## Cut Order Under Time Pressure

Evaluation is never cut — it is the deliverable. Cut in this order:
1. Confusion matrix for the `base` run (keep `tuned`)
2. `docs/CONCEPTS.md` sections 9–10
3. README "Approach" prose (keep the results tables and limitations)
4. Hugging Face Hub push of the adapter
