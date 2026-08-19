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

    # Average over the classes that actually occur in the gold set. Two reasons:
    #   1. A class with no gold examples has an UNDEFINED F1, not a zero one.
    #      Passing the full label list would force it to 0 (zero_division=0) and
    #      drag the macro average down for a reason unrelated to model quality.
    #   2. It excludes INVALID, so an unparseable prediction counts as a miss on
    #      the true class rather than inventing a 23rd class of its own.
    # On our real test set all 22 labels are present (support 7-10 each), so
    # this is identical to passing the full list — it only matters if a future
    # test split omits a rare condition.
    evaluated_labels = sorted(set(normalized_golds))
    macro_f1 = f1_score(
        normalized_golds,
        extracted,
        labels=evaluated_labels,
        average="macro",
        zero_division=0,
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
