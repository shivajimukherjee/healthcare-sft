"""Evaluate the base model and the tuned model on identical held-out data.

WHY THE BASELINE IS NOT OPTIONAL
--------------------------------
"87% accuracy" means nothing on its own. "62% -> 87% on the same 212 held-out
examples, same prompts, same decoding" is a result. This script therefore runs
in two modes over one code path, so the two runs cannot accidentally differ:

    python -m src.evaluate --no-adapter --tag base
    python -m src.evaluate --adapter adapters/qwen-healthcare-lora --tag tuned

Everything is held constant between the runs except the adapter: same prompts
(built by src.prompts), same 4-bit quantization, same greedy decoding, same
test files. Decoding is greedy (do_sample=False) so no sampling luck enters the
comparison and re-running reproduces the numbers exactly.
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
    INVALID,
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
    # Left padding is required for batched generation. With right padding the
    # model would continue from pad tokens rather than from the end of the
    # prompt, and short prompts in a batch would produce garbage.
    tokenizer.padding_side = "left"

    outputs = []
    for start in range(0, len(messages_list), batch_size):
        chunk = messages_list[start : start + batch_size]
        prompts = [render_prompt(m, tokenizer) for m in chunk]
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
        done = min(start + batch_size, len(messages_list))
        print(f"  generated {done}/{len(messages_list)}", flush=True)
    return outputs


def save_confusion_matrix(golds, preds, labels, path, tag):
    """Confusion matrix over the 22 labels PLUS an explicit invalid column.

    The invalid column is not decoration. sklearn drops any sample whose
    prediction is absent from `labels`, so plotting with the 22 labels alone
    would silently discard most of the BASE model's 212 examples and present a
    matrix that looks fine while describing a fraction of the data. Adding
    INVALID keeps every sample on the plot and turns the untuned model's
    failure mode into a visible column.
    """
    columns = list(labels) + [INVALID]
    fig, ax = plt.subplots(figsize=(14, 12))
    ConfusionMatrixDisplay.from_predictions(
        [normalize_label(g) for g in golds],
        [extract_label(p, labels) for p in preds],
        labels=columns,
        display_labels=[*labels, "INVALID"],
        xticks_rotation="vertical",
        ax=ax,
        colorbar=False,
    )
    ax.set_title(f"Classification confusion matrix — {tag}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_qualitative(preds, refs, path, tag, n=5):
    """Side-by-side samples.

    ROUGE cannot tell you that a summary is well-formed but factually wrong, or
    badly formed but right. Reading five examples is worth a metric.
    """
    with Path(path).open("w") as handle:
        handle.write(f"# Qualitative examples — {tag}\n\n")
        for i in range(min(n, len(preds))):
            handle.write(f"## Example {i + 1}\n\n")
            handle.write(f"**Model output**\n\n```\n{preds[i]}\n```\n\n")
            handle.write(f"**Reference**\n\n```\n{refs[i]}\n```\n\n---\n\n")


def _load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--no-adapter", action="store_true")
    parser.add_argument("--tag", required=True, help="'base' or 'tuned'")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--limit", type=int, default=None, help="Evaluate N examples only (smoke test)."
    )
    args = parser.parse_args()

    data_dir, results_dir = Path(args.data_dir), Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    labels = json.loads((data_dir / "labels.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = load_model(None if args.no_adapter else args.adapter)

    # --- classification -----------------------------------------------------
    clf_rows = _load_jsonl(data_dir / "test_classification.jsonl")
    if args.limit:
        clf_rows = clf_rows[: args.limit]
    # Drop the assistant turn: that is the answer we are trying to predict.
    clf_prompts = [row["messages"][:-1] for row in clf_rows]
    clf_golds = [row["messages"][-1]["content"] for row in clf_rows]

    print(f"[{args.tag}] classification: {len(clf_prompts)} examples", flush=True)
    # 16 new tokens is ample for a label and stops the base model rambling
    # for a paragraph, which would only waste time.
    clf_preds = generate(model, tokenizer, clf_prompts, max_new_tokens=16)
    clf_metrics = classification_metrics(clf_preds, clf_golds, labels)
    print(f"[{args.tag}] {clf_metrics}", flush=True)

    # --- summarization ------------------------------------------------------
    summ_rows = _load_jsonl(data_dir / "test_summarization.jsonl")
    if args.limit:
        summ_rows = summ_rows[: args.limit]
    summ_prompts = [row["messages"][:-1] for row in summ_rows]
    summ_refs = [row["messages"][-1]["content"] for row in summ_rows]

    print(f"[{args.tag}] summarization: {len(summ_prompts)} examples", flush=True)
    summ_preds = generate(
        model, tokenizer, summ_prompts, max_new_tokens=256, batch_size=4
    )
    summ_metrics = rouge_metrics(summ_preds, summ_refs)
    summ_metrics["section_adherence"] = section_adherence(summ_preds)
    summ_metrics["n"] = len(summ_preds)
    print(f"[{args.tag}] {summ_metrics}", flush=True)

    # --- persist ------------------------------------------------------------
    # Read-modify-write so the base and tuned runs accumulate into one file
    # rather than overwriting each other.
    metrics_path = results_dir / "metrics.json"
    all_metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    all_metrics[args.tag] = {
        "classification": clf_metrics,
        "summarization": summ_metrics,
    }
    metrics_path.write_text(json.dumps(all_metrics, indent=2))

    save_confusion_matrix(
        clf_golds,
        clf_preds,
        labels,
        results_dir / f"confusion_matrix_{args.tag}.png",
        args.tag,
    )
    save_qualitative(
        summ_preds, summ_refs, results_dir / f"qualitative_{args.tag}.md", args.tag
    )
    # Classification samples too — this is where task interference shows up,
    # e.g. a four-section note emitted in answer to a CLASSIFY prompt.
    save_qualitative(
        clf_preds,
        clf_golds,
        results_dir / f"qualitative_classification_{args.tag}.md",
        args.tag,
    )

    print(f"[{args.tag}] wrote results to {results_dir}", flush=True)


if __name__ == "__main__":
    main()
