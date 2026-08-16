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
