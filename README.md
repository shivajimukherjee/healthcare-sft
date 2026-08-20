# Healthcare LoRA SFT

A single LoRA adapter on `Qwen2.5-1.5B-Instruct` that performs two clinical NLP
tasks: symptom-to-condition classification and doctor–patient dialogue
summarization. Trained with QLoRA on a free Colab T4 in about 30 minutes.

> **Decision-support demonstration only. Not a medical device, not medical
> advice.** Outputs are unverified and would require review by a qualified
> clinician in any real setting.

## Results

Held-out test sets. Both models receive byte-identical prompts, the same 4-bit
quantization and the same greedy decoding — the only difference is whether the
adapter is attached.

### Classification — 212 held-out examples, 22 conditions

| Metric | Base | Tuned | Δ |
|---|---|---|---|
| Accuracy | 30.7% | 92.0% | **+61.3** |
| Macro-F1 | 25.8% | 91.6% | **+65.8** |
| Invalid-label rate | 15.1% | 0.0% | **−15.1** |

### Summarization — 100 held-out dialogues

| Metric | Base | Tuned | Δ |
|---|---|---|---|
| ROUGE-1 | 0.480 | 0.715 | **+0.235** |
| ROUGE-2 | 0.305 | 0.563 | **+0.258** |
| ROUGE-L | 0.418 | 0.653 | **+0.234** |
| Section adherence | 96.0% | 100.0% | **+4.0** |

Raw numbers: [`results/metrics.json`](results/metrics.json). Confusion matrices:
[`results/confusion_matrix_base.png`](results/confusion_matrix_base.png) and
[`results/confusion_matrix_tuned.png`](results/confusion_matrix_tuned.png).
Side-by-side outputs: [`results/qualitative_tuned.md`](results/qualitative_tuned.md).

### Reading these numbers honestly

**The classification gain is reliability, not medical knowledge.** Fine-tuning
851 examples does not install clinical understanding into 1.5B frozen
parameters. The base model's first test prediction was `gastritis` against a
reference of `peptic ulcer disease` — a sensible answer from the whole of
medicine, scored against a closed 22-label vocabulary. SFT taught the
vocabulary and the output contract. That is what SFT reliably teaches.

**Section adherence was never the differentiator.** The base model already
scored 96%, because the system prompt states the four required headers
explicitly. Note also that only **88%** of the *gold* reference notes contain
all four headers, so the tuned model's 100% does not mean superhuman — it means
it followed the instructed format more consistently than its own noisy training
targets did. The real summarization gain shows up in ROUGE, where the tuned
model dropped the base model's markdown decoration (`**Symptoms:**`) and matched
the reference format exactly.

**No task interference.** One adapter serves both tasks, and the tuned model
still answers `CLASSIFY` prompts with a bare label rather than a clinical note.
This was checked, not assumed — see `results/qualitative_classification_tuned.md`.

## Approach

**Base model: `Qwen/Qwen2.5-1.5B-Instruct`.** Small enough to QLoRA-train on a
free T4 and to run locally on an 8 GB Mac; strong enough at instruction
following to make a fair baseline. Instruct rather than Base, because at ~2,000
examples there is nowhere near enough data to teach chat formatting from
scratch. Qwen2.5 rather than Qwen3 deliberately: Qwen3's hybrid thinking mode
emits `<think>` blocks by default, which complicates both loss masking and
output parsing for no benefit here.

**QLoRA configuration**

| Setting | Value | Why |
|---|---|---|
| Quantization | 4-bit NF4, double quant | 3 GB → ~0.9 GB; safe because base weights are frozen |
| `r` | 16 | 18.5M trainable params = **1.18%** of the model |
| `lora_alpha` | 32 | `alpha/r` is a scale factor, so `r` can change without re-tuning the LR |
| `target_modules` | `q,k,v,o,gate,up,down` | attention **and** MLP; attention-only measurably underperforms |
| Learning rate | 2e-4, cosine, 22 warmup steps | ~10× a full-fine-tune LR, safe because `B` starts at zero |
| Epochs | 3 | 765 optimizer steps at an effective batch of 8 |
| Precision | fp16 | T4 is Turing — no bf16 |
| Attention | `sdpa` | flash-attention-2 requires Ampere+ |
| Optimizer | `adamw_torch` | **not** `paged_adamw_8bit` — see below |

**One adapter, two tasks.** Behaviour is selected at inference time by a task
tag in the prompt (`[TASK: CLASSIFY]` / `[TASK: SUMMARIZE]`), not by a switch in
the weights. This is how instruct models are made in the first place. The
training file is shuffled so every batch mixes both tasks, and the two are
scored separately so interference would be visible.

**Generative classification, not a classification head.** The model writes the
label as text rather than selecting from a softmax over 22 classes. This keeps
one objective, one adapter and one inference path for both tasks. The cost is
that invalid output is possible, which is precisely why invalid-label rate is a
headline metric rather than an afterthought.

**Loss masking written by hand.** No TRL. `src/tokenization.py` builds the
`labels` array and sets prompt positions to `-100` explicitly. TRL's
`SFTTrainer` automates this, but its API churns between releases and the
mechanism is thirty lines worth being able to explain. That choice paid off when
`transformers` 5.x changed `apply_chat_template`'s return type — the breakage was
visible and fixable rather than buried in a library.

**A bug worth recording.** The standard QLoRA recipe specifies
`optim="paged_adamw_8bit"`. It raised `CUDA error: an illegal memory access`
five steps into training. The paged optimizer exists for full fine-tuning, where
optimizer state is many gigabytes; at 18.5M trainable parameters it saves 111 MB
— 0.7% of a T4 — in exchange for a fragile unified-memory code path. Plain
`adamw_torch` is correct at this scale.

## Data

| Dataset | Rows | Columns | License |
|---|---|---|---|
| [`gretelai/symptom_to_diagnosis`](https://huggingface.co/datasets/gretelai/symptom_to_diagnosis) | 853 train / 212 test | `input_text`, `output_text` | Apache 2.0 |
| [`har1/MTS_Dialogue-Clinical_Note`](https://huggingface.co/datasets/har1/MTS_Dialogue-Clinical_Note) | 1,301 train | `dialogue`, `section_text` | MIT |

Both are public and de-identified. No PHI is handled anywhere in this project.

**Split discipline.** The classification corpus ships an author-provided test
split, used untouched. The summarization corpus ships **no** held-out split, so
100 rows are carved out with a fixed seed — a weaker test set than an
independent one, since it shares annotators and distribution, but never trained
on. The 22-label vocabulary is derived from the **training** split only;
deriving it from the full dataset would leak test information into every prompt.

**Contamination check.** Both corpora contain exact duplicates — including one
`input_text` present in gretelai's own train *and* test splits. Three training
rows appearing in a test set were dropped, and the test sets verified clean
afterwards.

**Length filter.** Examples over 1,024 tokens are dropped, never truncated — 11
summarization rows. Truncating a dialogue while keeping its full summary would
train the model to state facts it cannot see, which is teaching hallucination.

Final training set: **2,040 examples** (851 classification, 1,189 summarization),
shuffled with seed 42.

## Reproducing

**Local** (data prep, metrics, tests, demo — no GPU):

```bash
uv venv --python 3.11
uv pip install -r requirements.txt
.venv/bin/python -m pytest              # 37 tests
.venv/bin/python -m src.data_prep       # rebuilds data/
```

**Training and evaluation** (Colab T4): open
[`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb), set the runtime to
T4 GPU, and run top to bottom. It clones this repo, installs
`requirements-train.txt`, rebuilds the data, runs a 10-example smoke test, trains
for ~30 minutes and evaluates both models.

```bash
python -m src.train
python -m src.evaluate --no-adapter --tag base
python -m src.evaluate --adapter adapters/qwen-healthcare-lora --tag tuned
```

**Local demo** (fp16, ~3 GB, fits an 8 GB Mac — `bitsandbytes` has no Apple
Silicon build, so no 4-bit locally):

```bash
.venv/bin/python -m src.infer --task classify \
  --text "I have a burning feeling when I urinate and I need to go constantly."

.venv/bin/python -m src.infer --task summarize \
  --text "Doctor: What brings you in today? Patient: My right knee has been aching for three weeks, worse on stairs."
```

Add `--no-adapter` to run the same prompt through the untuned base model.

## Repository layout

```
src/prompts.py        Prompt construction — the single source of truth for what
                      both models see. Zero third-party dependencies.
src/tokenization.py   Chat templating and completion-only loss masking.
src/data_prep.py      Downloads both corpora, filters, mixes, splits, writes data/.
src/metrics.py        Accuracy, macro-F1, invalid rate, ROUGE, section adherence.
src/train.py          QLoRA training. Runs on Colab.
src/evaluate.py       Base vs tuned over one code path. Runs on Colab.
src/infer.py          Single-example CLI for a live demo. Runs locally in fp16.
tests/                37 tests, all CPU, no network beyond the tokenizer.
notebooks/            The Colab training notebook.
docs/CONCEPTS.md      Full explanation of every technique used here.
data/                 Generated by src.data_prep.
adapters/             The trained adapter (70 MB).
results/              metrics.json, confusion matrices, qualitative samples.
```

## Limitations

- **Not a medical device.** No PHI is handled; both datasets are public and
  de-identified.
- **SFT taught output format and task behaviour, not medical knowledge.** A
  1.5B model's clinical knowledge comes from pretraining and is not reliable.
- **22 conditions is a closed toy label space.** Real triage is open-set and
  spans thousands of ICD codes, most of them rare.
- **ROUGE measures word overlap, not faithfulness.** It cannot see negation:
  "no evidence of pneumonia" and "evidence of pneumonia" score almost
  identically. A fluent, well-structured, factually wrong summary scores well.
  Hallucinated findings in a clinical note are a patient-safety issue, not a UX
  issue.
- **No validation split.** Hyperparameters were fixed in advance from published
  LoRA defaults and the test set was read exactly once. That keeps the reported
  numbers honest, but it means no early stopping and no checkpoint selection.
- **The summarization test set is self-carved**, so it shares annotators and
  distribution with the training data and is an easier test than a truly
  independent one.
- **Any real deployment needs abstention and human-in-the-loop review.**

## Why a small local model

Clinical data frequently cannot leave the building. A 1.5B model that runs
on-premise — or on a laptop — is a different product from an API call to a
frontier model, regardless of benchmark scores.

The adapter architecture is the commercially interesting part. One 3 GB base
model in memory, with a folder of 70 MB adapters swapped per request: one per
client, specialty, or note format. Twenty behaviours cost 3 GB + 20 × 70 MB,
not 20 × 3 GB. `adapter_config.json` names the base model it patches, so an
adapter is a versioned diff rather than another model to host.

## What I would do next

- **Constrained decoding** for classification, so invalid output is impossible
  by construction rather than merely unobserved.
- **A proper validation split**, enabling early stopping and checkpoint
  selection without touching the test set.
- **LLM-as-judge** for summaries, scoring faithfulness and hallucination —
  the things ROUGE structurally cannot see.
- **Calibration and abstention**, so the model can route to a human instead of
  being forced to guess.
- **Per-client adapters**, exercising the hot-swap property properly.

Full detail in [`docs/CONCEPTS.md`](docs/CONCEPTS.md) §10.
