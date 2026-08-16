# Healthcare LoRA SFT — Design

**Date:** 2026-08-17
**Status:** Approved
**Context:** Interview take-home for a Sutherland follow-up technical round.

---

## 1. Goal

Fine-tune a small open-weights LLM with LoRA so that a single adapter performs two
healthcare NLP tasks:

1. **Classification** — map a free-text symptom description to one of 22 conditions.
2. **Summarization** — turn a doctor–patient dialogue into a structured clinical note.

The deliverable is a GitHub repository that a technical interviewer can read in ten
minutes and come away convinced the author understands the full SFT pipeline: data,
prompt format, PEFT mechanics, training configuration, evaluation against a baseline,
and the limitations of the result.

**Success is defined by the evaluation, not the model.** A modest quality gain that is
honestly measured against a base-model baseline is the goal. A large claimed gain with
no baseline is a failure.

### Non-goal, stated explicitly

This project does not produce a diagnostic tool. LoRA SFT teaches *format and task
behavior*; it does not install medical knowledge. Every claim in the README must respect
that distinction.

---

## 2. Scope

**In scope**

- One base model, one multi-task LoRA adapter, two tasks.
- Two public, de-identified, permissively licensed datasets.
- One training run on a free Google Colab T4.
- Evaluation of both tasks, base model vs. tuned model, on held-out data.
- A repository with runnable scripts, a training notebook, results, and a written
  concepts explainer.

**Out of scope** (explicitly cut for the 1–2 day timebox)

- Hyperparameter sweeps or ablations
- RAG, agents, tool use
- A Gradio or web demo
- Models above 3B parameters
- BERTScore, LLM-as-judge evaluation
- Model merging or quantized export (GGUF)
- Deployment or serving infrastructure

---

## 3. Datasets

Both are public, already de-identified, and permissively licensed. No PHI is handled at
any point in this project — a claim the README will make explicitly.

| Task | Dataset | Splits | Fields | License |
|---|---|---|---|---|
| Classification | [`gretelai/symptom_to_diagnosis`](https://huggingface.co/datasets/gretelai/symptom_to_diagnosis) | 853 train / 212 test | `input_text` → `output_text` | Apache 2.0 |
| Summarization | [`har1/MTS_Dialogue-Clinical_Note`](https://huggingface.co/datasets/har1/MTS_Dialogue-Clinical_Note) | 1201 train / 100 validation | `dialogue` → `section_text` | MIT |

**Split discipline.** The classification set ships with an author-provided test split;
we use it unmodified so the evaluation is not self-graded. The summarization set's
100-row `validation` split becomes our test set. Nothing from either test set is seen
during training.

**Label space.** The 22 canonical labels are derived from the *training* split only and
written to `data/labels.json`. Deriving them from the test split would leak.

**Length filtering.** MTS dialogues run up to ~8,900 characters. Examples whose rendered
prompt exceeds the training sequence budget are dropped rather than silently truncated —
truncation would produce targets that reference content the model cannot see, teaching it
to hallucinate. The number of dropped examples is recorded and reported.

---

## 4. Model and approach

**Base model: `Qwen/Qwen2.5-1.5B-Instruct`**

Rationale: Apache 2.0 and ungated (no access request to wait on); mature `transformers` /
`peft` / `trl` support; no hybrid-thinking mode, so the chat template will not inject
`<think>` blocks that would complicate loss masking and output parsing; and small enough
to run inference locally on the author's 8 GB M1 for a live demo.

Qwen3-1.7B was considered and rejected: its thinking mode is a debugging risk that the
timebox does not accommodate, and the tradeoff is itself a defensible interview answer.

**Method: QLoRA — 4-bit frozen base + LoRA adapters**

**One adapter, both tasks.** An adapter is a weight diff applied to the whole model, not
a task-bound artifact; behavior is selected at inference time by the prompt. Multi-task
SFT is the normal case — the base Instruct model was itself produced that way. Two
adapters would be chosen for operational reasons (independent release cycles, per-client
customization), none of which apply here.

**Generative classification, not a classification head.** The model writes the label as
text rather than using `AutoModelForSequenceClassification`. This keeps both tasks under
one objective, one adapter, and one inference path. The cost is that invalid outputs are
possible — which is why *invalid-label rate* becomes a headline metric rather than a
triviality.

---

## 5. Data pipeline and prompt format

`src/data_prep.py` loads both datasets, renders each example into Qwen's ChatML chat
template, interleaves them, shuffles with a fixed seed, and writes JSONL to `data/`.

Each record is a `messages` list of system / user / assistant turns.

**Classification prompt**

- *System:* clinical triage assistant framing, plus **the full list of 22 valid labels**.
- *User:* `[TASK: CLASSIFY]` tag followed by the symptom text.
- *Assistant:* the normalized label, nothing else.

The label list appears in both training and evaluation prompts, identically for base and
tuned models. This keeps the baseline fair — a base model asked to pick from an unstated
label set would fail for the wrong reason. It also reframes what is being learned: not
memorizing the label space, but reliably mapping symptom language onto it.

**Summarization prompt**

- *System:* clinical documentation assistant framing; emit exactly four sections —
  `Symptoms`, `Diagnosis`, `History of Patient`, `Plan of Action` — using `N/A` where a
  section was not discussed.
- *User:* `[TASK: SUMMARIZE]` tag followed by the dialogue.
- *Assistant:* the reference `section_text`.

**No safety disclaimer in the training target.** A disclaimer in the assistant turn would
be learned as part of the output and would pollute ROUGE against the references. The
disclaimer belongs in the inference wrapper and the README, not in the weights.

**Task tags** are cheap insurance. The two system prompts and output shapes are already
highly distinct; the explicit tag makes the task boundary trivially learnable and costs a
handful of tokens.

**Mixture balance.** 853 classification and 1201 summarization examples are naturally
near-balanced. They are interleaved and shuffled so most batches contain both tasks.

**Loss masking.** Loss is computed on assistant tokens only. Training on the prompt
tokens would spend adapter capacity learning to reproduce a system prompt that is
identical in every example.

---

## 6. Training configuration

`src/train.py`, driven from `notebooks/train_colab.ipynb`.

| Setting | Value | Why |
|---|---|---|
| Quantization | 4-bit NF4, double quant | Fits the frozen base in T4 memory |
| Compute dtype | **fp16** | T4 is Turing — **no bf16 support** |
| Attention | `sdpa` | **flash-attention-2 requires Ampere+**; unavailable on T4 |
| LoRA rank `r` | 16 | Enough capacity for two tasks at this data scale |
| `lora_alpha` | 32 | Convention `α = 2r`; a scaling factor, not extra capacity |
| `lora_dropout` | 0.05 | Light regularization on a small dataset |
| `target_modules` | `q,k,v,o,gate,up,down_proj` | Attention *and* MLP; MLP-only targeting underperforms |
| Learning rate | 2e-4, cosine, 3% warmup | LoRA tolerates ~10× a full-FT learning rate |
| Epochs | 3 | More overfits at ~2k examples |
| Batch | 2 × grad-accum 4 = **8 effective** | Fits T4 memory |
| `max_seq_length` | 1024 | Covers most dialogues; drives the length filter |
| Optimizer | `paged_adamw_8bit` | Reduces optimizer memory |
| Gradient checkpointing | on | Trades compute for memory |
| Seed | 42 | Reproducibility |

Expected: ~770 optimizer steps, roughly 20–40 minutes on a T4. If throughput is worse
than expected, the first lever is dropping to 2 epochs, then reducing `max_seq_length`
to 768.

The adapter is saved to `adapters/qwen-healthcare-lora/` and optionally pushed to the
Hugging Face Hub.

---

## 7. Evaluation

`src/evaluate.py`, runnable with or without the adapter. **The base-model baseline on the
identical test set and identical prompts is mandatory** — it is the single thing that
converts numbers into a result. Decoding is greedy (`do_sample=False`) for both models so
the comparison is deterministic.

**Classification** — 212 held-out examples

- Exact-match accuracy after normalization (lowercase, strip)
- **Macro-F1** — the label distribution is uneven, so macro matters more than micro
- **Invalid-label rate** — predictions matching none of the 22 labels. Expected to be
  substantial for the base model and near zero after tuning; this is the clearest single
  demonstration of what SFT actually does
- Confusion matrix, saved as a PNG for each model

**Summarization** — 100 held-out examples

- ROUGE-1 / ROUGE-2 / ROUGE-L
- **Section-adherence rate** — fraction of outputs containing all four required headers.
  This is the format-learning signal, and ROUGE alone will not show it
- 3–5 side-by-side qualitative examples (base / tuned / reference) written to
  `results/qualitative_examples.md`

ROUGE is a weak proxy for summary quality. The README will say so and lean on the
qualitative examples and section adherence alongside it.

**Task-interference check.** Per-task metrics double as the interference detector. If
classification degrades or the model emits summaries when asked to classify, the ordered
remedies are: raise `r` to 32, rebalance the mixture, and only then split into two
adapters. Whatever is observed gets reported.

All metrics are written to `results/metrics.json` and rendered as a README table.

---

## 8. Repository structure

```
healthcare-sft/
├── README.md                     # problem, approach, results table, limitations
├── requirements.txt              # pinned versions
├── data/
│   ├── labels.json               # 22 canonical labels (from train split only)
│   ├── train.jsonl               # mixed, shuffled, chat-formatted
│   ├── test_classification.jsonl
│   └── test_summarization.jsonl
├── src/
│   ├── data_prep.py              # load → format → filter → mix → split
│   ├── train.py                  # QLoRA SFT
│   ├── evaluate.py               # both tasks, base vs tuned
│   └── infer.py                  # load base + adapter, single prediction, disclaimer
├── notebooks/
│   └── train_colab.ipynb         # end-to-end runnable on a free T4
├── adapters/
│   └── qwen-healthcare-lora/     # adapter_config.json + adapter_model.safetensors
├── results/
│   ├── metrics.json
│   ├── confusion_matrix_base.png
│   ├── confusion_matrix_tuned.png
│   └── qualitative_examples.md
└── docs/
    ├── CONCEPTS.md               # standalone SFT/LoRA/QLoRA explainer
    └── superpowers/specs/        # this document
```

---

## 9. Pedagogical requirement

The author is new to SFT and must be able to defend every line in a live interview.
Explanation is therefore a **first-class deliverable**, not a documentation afterthought:

- Every source file opens with a docstring covering *what* it does and *why* the approach
  was chosen.
- Inline comments explain non-obvious ML decisions at their site: why `B` is zero-
  initialized, why loss is masked to completions, why fp16 rather than bf16, why a paged
  optimizer, why macro-F1.
- `docs/CONCEPTS.md` is a standalone explainer — SFT in the training pipeline, the LoRA
  decomposition, what an adapter file is, QLoRA, the evaluation metrics and their
  weaknesses. Written to be revised from the night before the interview.
- The notebook carries a markdown cell before each code cell explaining the step.
- Each component is explained conversationally as it is built.

Code that works but cannot be explained is a failed deliverable for this project.

---

## 10. Healthcare framing and limitations

The README must state plainly:

- **Not a medical device.** Decision-support demonstration only; not for clinical use.
- **No PHI.** Both datasets are public and de-identified; no patient data was handled.
- **What was actually learned** — output format and task behavior, not medical knowledge.
  A 1.5B model's clinical knowledge comes from pretraining and is not reliable.
- **Failure modes** — hallucinated findings in a clinical note are a patient-safety issue,
  not a UX issue. Abstention and human-in-the-loop review are required in any real
  deployment.
- **Narrow label space** — 22 conditions is a toy space; real triage is open-set.
- **Why a small local model** — clinical data often cannot leave the building. An on-prem
  1.5B model with swappable adapters is a genuine architectural argument, and the strongest
  framing for this work.

---

## 11. Risks and mitigations

| Risk | Mitigation |
|---|---|
| TRL/PEFT API churn breaks the training call | Pin versions in `requirements.txt`; verify the loss-masking path on a 10-example smoke run before the full run |
| Colab session dies mid-training | Save checkpoints each epoch; download or push the adapter immediately on completion |
| T4 out-of-memory | Batch size 1, `max_seq_length` 768, keep gradient checkpointing on |
| Task interference | Detected by per-task metrics; ordered remedies in §7 |
| Long dialogues lose content | Filter rather than truncate; report the dropped count |
| ROUGE gain is unimpressive | Section-adherence rate and qualitative examples carry the summarization story |
| Time overrun | Evaluation is never cut. The Hub push, the second confusion matrix, and README polish are the cut order |

---

## 12. Timeline

| | Work |
|---|---|
| **Day 1 AM** | Repo scaffold, `data_prep.py`, inspect formatted examples by eye, smoke-test the training call on 10 examples |
| **Day 1 PM** | Full Colab training run, save adapter, sanity-check a few generations |
| **Day 2 AM** | `evaluate.py`, run base and tuned, produce metrics and confusion matrices |
| **Day 2 PM** | `CONCEPTS.md`, README with results table and limitations, qualitative examples |

**Definition of done:** the repo clones and runs; `results/metrics.json` contains base and
tuned numbers for both tasks; the README states results and limitations honestly; and the
author can explain any file in it without reading ahead.
