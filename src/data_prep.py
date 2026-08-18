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
untouched so the evaluation is not self-graded.

The summarization dataset ships ONLY a train split (1301 rows) — the plan
assumed a `validation` split that does not exist. We therefore carve our own
100-row test set with a fixed seed. This is weaker than an author-provided
split (same distribution, same annotators) and the README says so, but the
rows are still never trained on, which is the property that matters.

Labels are derived from the TRAINING split only — deriving them from test
would leak test information into every prompt.

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
N_SUMMARIZATION_TEST = 100


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

    # The summarization corpus has no held-out split of its own, so we make one.
    # Seeded, so the same 100 rows are held out on every machine and every run —
    # an unseeded split would silently leak different rows into training each
    # time and make the reported numbers unreproducible.
    summ_split = summ["train"].train_test_split(
        test_size=N_SUMMARIZATION_TEST, seed=SEED
    )

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

    for row in summ_split["train"]:
        # .strip() is load-bearing, not cosmetic. One row in this corpus has a
        # section_text beginning with "\n". The prompt already ends with "\n",
        # and BPE merges the two into a single "\n\n" token — so the prompt's
        # tokens stop being a prefix of the full sequence and every label in
        # that example shifts by one. build_training_example raises on it.
        # Leading/trailing whitespace in a target is meaningless anyway.
        messages = build_summarization_messages(
            row["dialogue"], answer=row["section_text"].strip()
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
                row["dialogue"], answer=row["section_text"].strip()
            ),
        }
        for row in summ_split["test"]
    ]

    # Remove any training row whose user turn also appears in a test set.
    # Both source corpora contain exact duplicates — gretelai ships one
    # input_text in BOTH its own author-provided train and test splits, and
    # MTS-Dialogue has 3 repeated dialogues, so any split can land copies on
    # either side. It is only ~1% of the test sets, but train/test contamination
    # inflates the headline number for free, so we drop it and report how much.
    test_user_turns = {
        record["messages"][1]["content"] for record in clf_test + summ_test
    }
    before = len(train_records)
    train_records = [
        record
        for record in train_records
        if record["messages"][1]["content"] not in test_user_turns
    ]
    print(f"dropped {before - len(train_records)} training rows found in a test set")

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
