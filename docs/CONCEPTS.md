# Concepts

A standalone explanation of every technique this project uses, and why each
choice was made. Written to be read start to finish before defending the work.

---

## 1. Where SFT sits

A modern chat model is built in three stages, and it helps to be precise about
what each one buys.

**Pretraining** runs next-token prediction over trillions of tokens of raw
text. This is where essentially all of the model's knowledge and linguistic
competence is acquired. It costs millions of dollars and produces a *base*
model: excellent at continuing text, useless at following instructions,
because nothing in the objective ever taught it that a question should be
answered rather than extended into more questions.

**Supervised fine-tuning** — this project — trains on (prompt, ideal response)
pairs. Mechanically it is the *same* next-token prediction. The only
difference is that the loss is computed on the response tokens alone, so the
gradient pressure lands entirely on "given this instruction, produce this
shape of answer". This is what converts a base model into an instruct model.
It takes hours and thousands of examples, not months and trillions of tokens.

**Preference tuning** (RLHF, DPO) comes third and ranks candidate responses
against each other, optimising for which of two answers a human prefers. It
sands off the remaining rough edges — tone, refusal behaviour, verbosity. This
project does not do it: it needs preference pairs we do not have, and SFT
alone is sufficient to demonstrate the format-following behaviour we are
after.

The one-sentence version: **pretraining teaches the model what it knows; SFT
teaches it how to answer; preference tuning teaches it which answer to
prefer.**

---

## 2. What SFT can and cannot teach

This distinction determines how every claim in this repository is worded, and
it is the question most worth being ready for.

**SFT reliably teaches format and task behaviour.** Answer with a bare label.
Use these four section headers. Do not add a preamble. Stop after the answer.
These are patterns the model can learn from a couple of thousand examples
because they are consistent, shallow, and present in every single one.

**SFT does not reliably teach new knowledge.** Fine-tuning 1.5 billion frozen
parameters plus an 18-million-parameter adapter on 851 symptom descriptions
does not install medical understanding the base model lacked. There is nowhere
near enough signal, and the base weights — where knowledge actually lives — do
not move at all.

So the honest reading of our headline number is this. Classification accuracy
rose from 0.307 to 0.920. That is **not** the model becoming a better
diagnostician. Qwen2.5-1.5B already carried enough medical association to link
"burning urination and frequency" with urinary tract infection; what it lacked
was the discipline to answer with exactly one label drawn from a fixed
vocabulary of twenty-two. Look at the base model's actual output on the first
test example:

```
base output:  gastritis
reference:    peptic ulcer disease
```

`gastritis` is a real gastrointestinal condition and a clinically sensible
guess. It simply is not in our label set. The base model was answering from
the whole of medicine while being graded against a closed vocabulary. SFT
taught it the vocabulary and the output contract, not the medicine.

If knowledge were genuinely required — a condition absent from pretraining, or
a hospital-specific protocol — the right tools would be retrieval-augmented
generation or continued pretraining on a domain corpus, not a small SFT run.

---

## 3. LoRA

Full fine-tuning updates every weight. For a 1.5B model that means holding the
weights, their gradients, and two Adam moments per parameter — roughly 24 GB
before activations. A 15 GB T4 cannot do it.

**Low-Rank Adaptation** starts from an empirical observation: the *update*
a fine-tune applies to a weight matrix tends to be low-rank, even though the
matrix itself is not. So instead of learning a full `d x d` update, learn two
thin matrices whose product approximates it.

Each targeted weight matrix `W` is frozen, and a trainable bypass is added
beside it:

```
h = Wx + (alpha / r) * B(Ax)
```

- `A` has shape `(r, d)` and is initialised from a random Gaussian.
- `B` has shape `(d, r)` and is initialised to **zero**.
- `r` is the rank — 16 here, against `d = 1536`.

Instead of `1536 x 1536 = 2.4M` parameters per matrix, we train
`2 x 16 x 1536 = 49K`. Across all targeted modules that is 18,464,768
trainable parameters against 1,562,179,072 total: **1.18%**.

**Why `B` starts at zero.** At step 0, `BA = 0`, so the bypass contributes
exactly nothing and the model behaves *identically* to the base model.
Training departs from a known-good state rather than a randomly perturbed one.
This is why LoRA tolerates a learning rate of `2e-4`, roughly ten times a
typical full-fine-tuning rate: a large step cannot damage the base model,
because the base model is not moving. If both `A` and `B` were random, the
model would start out corrupted and the first phase of training would be spent
undoing that.

**What `r` and `alpha` each control.** `r` is capacity: how expressive the
update can be. `alpha` appears only in the scaling factor `alpha / r`, which
multiplies the adapter's output. Because it is a *ratio*, changing `r` does
not force you to re-tune the learning rate — doubling `r` while keeping
`alpha = 2r` leaves the effective scale unchanged. That decoupling is the
entire reason `alpha` exists. The convention `alpha = 2r` is what this project
uses: `r=16, alpha=32`.

**Why attention and MLP.** We target `q_proj, k_proj, v_proj, o_proj`
(attention) *and* `gate_proj, up_proj, down_proj` (MLP). The original LoRA
paper adapted attention only, and that remains a common shortcut, but later
work found it measurably underperforms — most of a transformer's parameters,
and much of its stored behaviour, live in the MLP blocks. Skipping them leaves
capacity on the table for no memory saving worth having.

---

## 4. What an adapter file actually is

Training produced this:

```
adapters/qwen-healthcare-lora/
├── adapter_config.json          1.2 KB
├── adapter_model.safetensors     70 MB   <- the A and B matrices
├── tokenizer.json                11 MB
└── ...
```

70 MB, against a base model of roughly 3 GB. `adapter_model.safetensors`
contains nothing but the `A` and `B` matrices for every targeted module.
(18.46M parameters at fp32 is 74 MB; stored at fp16 it would be half that.)

**An adapter is a diff, not a model.** It is meaningless on its own, and
`adapter_config.json` says so explicitly:

```json
"base_model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct",
"r": 16,
"lora_alpha": 32,
"target_modules": ["gate_proj","k_proj","up_proj","q_proj","o_proj","down_proj","v_proj"]
```

Loading it against a different base — or the same base at a different revision
— produces garbage or an outright shape error.

**Hot-swapping** is the practical payoff. A serving process holds one 3 GB
base model in memory and attaches whichever 70 MB adapter a request needs.
Twenty specialised behaviours cost 3 GB + 20 x 70 MB, not 20 x 3 GB. For a
services company that means one deployed base model with a per-client adapter
each, rather than twenty fine-tuned models to host and version.

**Merging.** `merge_and_unload()` folds `(alpha/r) * BA` directly into `W`,
yielding an ordinary model with zero inference overhead — no extra matmuls in
the forward pass. The cost is that you lose swappability: the result is a
single 3 GB model that does one thing. Merge when you serve one behaviour and
care about latency; keep the adapter separate when you serve many.

Unmerged LoRA does add a small inference cost — two extra thin matrix
multiplies per adapted layer — which is usually a few percent and often
invisible next to memory bandwidth limits.

---

## 5. QLoRA

QLoRA is LoRA with the frozen base model quantized to 4 bits.

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)
```

**NF4** — "4-bit NormalFloat" — is a data type whose quantization levels are
spaced according to a normal distribution rather than uniformly. Neural
network weights are approximately normally distributed, so NF4 places more
resolution where the values actually are, and loses less information than
plain int4 at the same bit width.

**Double quantization** quantizes the quantization constants themselves. Each
block of weights needs a scale factor stored alongside it; those scales are
themselves compressed. It saves roughly 0.4 bits per parameter — small, but
free.

**Why quantizing the base is safe here.** The base weights are *frozen*. They
are only ever read, never updated, so quantization error cannot accumulate
across steps the way it would if we were also training them. The trainable
LoRA matrices stay in full precision. Weights are dequantized on the fly for
each matrix multiply, which costs some speed and buys a 3x memory reduction:
about 3 GB down to about 0.9 GB.

That reduction is what puts a 1.5B fine-tune on a free T4 at all.

---

## 6. Completion-only loss masking

Implemented by hand in `src/tokenization.py`. This is the mechanical heart of
SFT and the part most tutorials hide inside a library call.

Every training example carries two parallel integer lists of identical length:

```
input_ids  = [151644, 8948, 198, ..., 1690, 91903, 151645, 198]   what the model reads
labels     = [  -100, -100, -100, ..., 1690, 91903, 151645, 198]   what it is graded on
```

`labels` is a copy of `input_ids` with every position belonging to the system
and user turns overwritten with `-100`. PyTorch's cross-entropy loss takes an
`ignore_index` parameter defaulting to `-100`; any position whose target is
`-100` contributes no loss and no gradient, and is excluded from the average.

**The critical distinction: the model still *sees* the masked tokens.** They
are untouched in `input_ids`, so attention reads them normally. They are
simply not graded. Masking affects the loss, not the visibility.

**Why it matters.** On a real example from this dataset, a classification
sequence is 217 tokens of which 5 are supervised. Without masking, 97.7% of
the gradient signal would come from reproducing a system prompt that is
byte-identical across all 2,040 training examples — adapter capacity spent
learning to recite our own instructions.

**How the boundary is found.** Not by searching the text for a marker. The
conversation is tokenized twice: once without the assistant turn
(`messages[:-1]`, 79 tokens) and once with it (83 tokens). The first length is
exactly how many positions to mask. A guard asserts that the shorter token
list really is a prefix of the longer one, because a tokenizer that merged
characters across that boundary would misalign every label without raising.

That guard earned its place. One row of the MTS-Dialogue corpus has a
`section_text` beginning with a newline; the prompt already ends with one, and
BPE merged the pair into a single `\n\n` token. The prompt was then no longer
a token-level prefix, and the example's labels would have shifted by one with
nothing to indicate it. Targets are now stripped in `data_prep.py`, and a test
pins the guard.

**Why by hand.** TRL's `SFTTrainer` automates this. It was written out
explicitly for two reasons: TRL's API changes frequently between releases and
is the fastest-moving dependency in this stack, and the mechanism is thirty
lines that are worth being able to explain. The second reason paid off — when
`transformers` 5.x changed `apply_chat_template` to return a `BatchEncoding`
rather than a list, the failure was visible and fixable rather than buried.

---

## 7. Multi-task behaviour in one adapter

One adapter serves both classification and summarization. The behaviour is
selected at inference time by the prompt, not by a switch in the weights:

```
[TASK: CLASSIFY]   + a symptom description  -> a bare label
[TASK: SUMMARIZE]  + a dialogue             -> a four-section note
```

There is nothing exotic here. A LoRA adapter is a weight diff applied to the
whole model, not a task-bound object, and the same forward pass runs in both
cases. This is also exactly how the base Instruct model was produced:
multi-task SFT over many instruction types is the normal case, not a trick.

**Task interference** is the real risk. Two tasks sharing one small set of
adapter parameters can degrade each other — most visibly, the model starts
emitting a four-section clinical note in response to a CLASSIFY prompt,
because summarization examples outnumber classification ones 1189 to 851.

Two defences. First, the training file is shuffled with a fixed seed so every
batch mixes both tasks; training all of one and then all of the other invites
the second to overwrite the first. Second, the tasks are scored *separately*
and qualitative samples are written for both, so interference is detected
rather than assumed away. A single blended metric would hide it completely.

In this run there was no interference: the tuned model answers CLASSIFY
prompts with bare labels and its invalid rate fell to zero. Had it risen after
tuning, the response would have been to increase `r` to 32 and retrain, giving
the two behaviours more capacity to coexist.

---

## 8. The metrics

**Classification.**

*Accuracy* is the fraction of examples answered correctly. Readable, and
misleading on its own when classes are unbalanced.

*Macro-F1* computes F1 per class and averages the classes without weighting
them by frequency. With twenty-two conditions this is the metric that matters,
because it is the one that notices a model quietly never predicting a rare
disease. A model that answers with the single most common label scores 0.047
accuracy but 0.004 macro-F1 on our test set — accuracy alone understates how
broken that is. One implementation note: the average is taken over classes
present in the gold set, because a class with no gold examples has an
*undefined* F1, not a zero one, and forcing it to zero would penalise a
perfect model for a gap in the split.

*Invalid rate* is the fraction of outputs that contain no valid label at all.
This is the clearest single demonstration of what SFT teaches, because it
isolates format compliance from medical correctness. Ours went from 0.151 to
0.000.

*Label extraction is deliberately generous.* `extract_label` accepts a label
embedded in a sentence, so "I believe this is malaria, based on the fever"
counts as `malaria`. That charity almost exclusively benefits the *baseline*,
which is the point: a baseline handicapped by output parsing would inflate the
reported improvement, and most of the gap would be punctuation rather than
capability.

**Summarization.**

*ROUGE-1/2/L* measure unigram, bigram, and longest-common-subsequence overlap
with the reference. ROUGE is genuinely weak: it rewards word overlap, not
faithfulness, and it cannot see negation. "Diagnosis: no evidence of
pneumonia" and "Diagnosis: evidence of pneumonia" share nearly every n-gram
and score almost identically while meaning opposite things. Negation is most
of clinical meaning, so ROUGE is never reported alone.

*Section adherence* is the fraction of outputs containing all four required
headers. ROUGE cannot see structure; this can. Its important caveat is that
only 88% of the *gold* notes contain all four headers, so 0.88 is the ceiling
the reference data itself achieves. The tuned model's 1.00 therefore does not
mean superhuman — it means it followed the instructed format more consistently
than its own noisy training targets did.

*Qualitative side-by-side examples* are written to `results/` for both models,
because neither ROUGE nor adherence can tell you that a summary is
well-formed and factually wrong.

**Why a baseline is mandatory.** "92% accuracy" is not a result. "30.7% to
92.0% on the same 212 held-out examples, same prompts, same greedy decoding,
same 4-bit quantization, differing only in whether the adapter is attached" is
a result. `src/evaluate.py` runs both through one code path so the two runs
cannot accidentally differ.

---

## 9. Hardware constraints

**The T4 is a Turing-generation GPU**, and two common settings do not work on
it:

- **No bf16.** bfloat16 arithmetic requires Ampere or newer. Training uses
  `fp16=True`. fp16 has a much narrower dynamic range than bf16, so small
  gradients can underflow to zero; a `GradScaler` multiplies the loss by a
  large constant before the backward pass and divides it out afterwards. This
  is why `nan` loss is the characteristic fp16 failure and why the training
  notes say to watch for it.
- **No flash-attention-2.** Also Ampere-and-newer. We use
  `attn_implementation="sdpa"`, PyTorch's built-in scaled dot-product
  attention, which is well optimised and works everywhere.

**Memory.** A T4 has 15 GB and everything must be resident at once: quantized
base weights (~0.9 GB), LoRA parameters (74 MB), their gradients (74 MB), Adam
moments (148 MB), and activations. `gradient_checkpointing=True` discards
intermediate activations and recomputes them during the backward pass, trading
roughly 30% speed for a large memory saving. Gradient accumulation runs four
micro-batches of two and sums their gradients before stepping, giving the
stability of a batch of eight at the memory cost of a batch of two.

**One config choice was wrong and got corrected.** The standard QLoRA recipe
specifies `optim="paged_adamw_8bit"`, and it raised
`CUDA error: an illegal memory access` inside bitsandbytes' `sync_gpu()` five
steps into training. The paged optimizer exists for *full* fine-tuning, where
optimizer state is many gigabytes and paging it to host RAM is the difference
between running and not. At 18.5M trainable parameters, Adam state is 148 MB
in fp32 against 37 MB paged-8-bit — a 111 MB saving, 0.7% of the card, bought
with a fragile CUDA unified-memory code path. Plain `adamw_torch` is the
correct choice at this scale, and the recipe was cargo-culted from tutorials
written for 7B+ models.

**Apple Silicon.** `bitsandbytes` has no Apple Silicon build, so 4-bit
quantization is unavailable locally and all training happens on Colab. This is
why `src/infer.py` loads the model in fp16 rather than reusing
`src/evaluate.py`'s quantized loader: 1.5B parameters at fp16 is about 3 GB,
which fits an 8 GB Mac comfortably. Everything that does not need a GPU —
prompts, tokenization, masking, metrics, data preparation — is unit tested
locally on CPU, which is why only the GPU-specific parts could fail on Colab.

---

## 10. What I would do next

**Constrained decoding for classification.** The invalid rate is already zero,
but nothing *structurally* prevents an invalid label. Constraining generation
to the 22-label vocabulary — or scoring all 22 and taking the argmax — would
make invalid output impossible by construction rather than merely unobserved,
and would also be faster than free generation.

**A proper validation split.** This project has train and test but no
validation set, so hyperparameters were fixed in advance from published LoRA
defaults and the test set was looked at exactly once. That is the honest way
to run without a validation set, but it means no early stopping and no
checkpoint selection. Carving 100 rows out of train would allow both without
contaminating the reported numbers.

**LLM-as-judge for summaries.** ROUGE cannot see negation or factual
faithfulness. A stronger model scoring each summary against its dialogue on
faithfulness, completeness, and hallucination would measure what actually
matters clinically. It is more expensive and introduces its own biases, which
is why it belongs in the next iteration rather than this one.

**Per-client adapters.** The hot-swapping property is the commercially
interesting one. One base model, one adapter per client or per specialty,
selected at request time — 70 MB of incremental storage per behaviour instead
of 3 GB.

**Calibration and abstention.** A triage model should be able to say "I do not
know". Exposing token-level confidence and abstaining below a threshold turns
a forced guess into a routed escalation, which is what a real clinical
workflow requires.

**A larger and more realistic label space.** Twenty-two conditions is a toy
vocabulary. Real triage spans thousands of ICD codes, most of them rare, which
turns this into an extreme multi-label problem where retrieval over a code
hierarchy is likely a better fit than generative classification.
