# prompts.py — explainer
> src/prompts.py @ 13c5466 · 99 lines · generated 2026-08-18

**How to read the evidence in this document.** Every substantive claim carries
one of three markers:

- **Observed** — printed by code that was actually run. §3 describes the probe.
- **Cited** — quoted from a docstring, spec, plan, commit message, or test in
  this repository, with the source named.
- **Inference** — my reading of the code. Reasonable, but unwritten and
  unverified.

Anything that is none of the three was cut rather than softened.

## 0. Orientation

This project's headline claim is "the fine-tuned model beats the untuned one on
the same test set". That claim is only worth anything if both models were asked
*exactly* the same question. This file is where the questions are written. It
builds the list of chat messages — a system instruction, a user turn, and
optionally the expected answer — for each of the project's two jobs:
classifying a symptom description into a condition, and summarizing a
doctor-patient dialogue into a structured note. It also holds one small text
cleaner used to compare a model's answer against the expected one fairly.

Nothing here talks to a model, a tokenizer, or a network. It is string
construction and nothing else, which is why it is trivially testable.

**Called by** — found with `grep -rn --exclude-dir={.venv,node_modules,.git,dist,build,__pycache__} "prompts" .`, not assumed:

- `tests/test_prompts.py` — the dedicated test module.
- `tests/test_tokenization.py` — imports `build_classification_messages` to
  build realistic fixtures (**Observed**: the grep found this import; it is also
  the fixture used in `docs/explainers/tokenization.md` §3).
- **Cited**, `docs/superpowers/plans/2026-08-17-healthcare-lora-sft.md`: three
  further callers are planned and **not yet written** — `data_prep.py` and
  `evaluate.py` are to import the message builders, and `metrics.py` is to
  import `normalize_label`. Neither `metrics.py` nor any script exists in the
  tree at the hash above.

**Calls out to:** nothing. There are no `import` statements in this file at all
— **Observed**, and unusual enough to be worth stating. Every dependency it has
is a Python builtin (`str.format`, `str.join`, `str.split`, `str.strip`).

**In one sentence:** it is the single place any prompt text in this project is
written, so training and evaluation cannot drift apart.

## 1. Inventory

| Symbol | Kind | Signature | Purpose |
|---|---|---|---|
| `CLASSIFY_TAG` | constant | `= "[TASK: CLASSIFY]"` | Marker prefixed to the user turn of a classification example, so the model can tell which job it is being asked to do. |
| `SUMMARIZE_TAG` | constant | `= "[TASK: SUMMARIZE]"` | The same, for summarization. Must differ from `CLASSIFY_TAG`; a test asserts it. |
| `normalize_label` | function | `normalize_label(text) -> str` | Reduces a condition name to a canonical form — lowercase, single-spaced, unpunctuated — so scoring is not thrown off by cosmetic differences. |
| `REQUIRED_SECTIONS` | constant | `= ["Symptoms", "Diagnosis", "History of Patient", "Plan of Action"]` | The four headers a clinical note must contain. Used to build the summarization system prompt, and (per its comment) to be shared with the scoring code. |
| `_CLASSIFY_SYSTEM` | constant | a template `str` with a `{labels}` slot | The classification system prompt, with a hole where the label list goes. Underscore-private. |
| `_SUMMARIZE_SYSTEM` | constant | a template `str` with a `{sections}` slot | The summarization system prompt, with a hole where the section headers go. Underscore-private. |
| `build_classification_messages` | function | `build_classification_messages(symptom_text, labels, answer=None) -> list[dict]` | Builds the 2- or 3-message list for one classification example. |
| `build_summarization_messages` | function | `build_summarization_messages(dialogue, answer=None) -> list[dict]` | The same for summarization. |

There is no package table: the file imports nothing (see §0).

## 2. How the code is written

Pass 1: a walk down the file through a **language** lens. Each construct is
named on its first appearance only.

### The module docstring (lines 1-22)

A string literal as the first statement of a file is the **module docstring**,
stored as `__doc__` and printed by `help()`. This one argues for the module's
existence rather than describing its API — it explains why prompts live in one
place at all.

### `CLASSIFY_TAG = "[TASK: CLASSIFY]"` (line 24)

A **module-level constant**. Python has no `const`; the `UPPER_SNAKE_CASE` name
is a convention meaning "do not reassign", enforced by other programmers rather
than by the language. Nothing stops `prompts.CLASSIFY_TAG = "x"` at runtime.

### `def normalize_label(text):` (line 28)

`def` **defines** a function without running it. `text` is a **parameter**; the
value supplied at a call site is the **argument**. No type annotations are used
anywhere in this file, and Python would not enforce them at runtime if they were.

#### Method chaining (line 39)

```python
cleaned = " ".join(text.strip().lower().split())
```

Read left to right, each call acting on the result of the previous one. Every
one of these returns a **new** string, because Python strings are **immutable** —
`text` itself is unchanged by any of them.

- `.strip()` with no argument removes leading and trailing **whitespace**.
- `.lower()` lowercases.
- `.split()` with no argument is the interesting one: it splits on *runs* of any
  whitespace — spaces, tabs, newlines — and discards empty pieces. That is what
  collapses `"urinary\n tract  infection"` into three words.
- `" ".join(list_of_strings)` then rebuilds one string with single spaces. Note
  the separator is the string you call the method *on*, which reads backwards
  the first time you meet it.

#### `.strip('.,;:!?"\'')` (line 40)

The same method with an argument, and it does **not** mean what most people
first assume. The argument is a **set of characters**, not a substring: `.strip`
removes any of those characters repeatedly from both ends until it meets one
that is not in the set. It never touches the middle of the string.

Inside the single-quoted literal `'.,;:!?"\''`, the `"` needs no escape but the
final `'` does — hence the backslash. The characters are just punctuation Python
has no opinion about.

### `REQUIRED_SECTIONS = [...]` (line 45)

A **list literal**. Unlike the string constants above, a list is **mutable** —
the `UPPER_CASE` name asks callers not to modify it, but nothing prevents
`REQUIRED_SECTIONS.append(...)` from anywhere in the program. §5 exercise 2 shows
what that costs.

### `_CLASSIFY_SYSTEM = """..."""` (lines 47-54)

Two constructs at once.

The **leading underscore** marks the name as internal to this module — a
convention, skipped by `from prompts import *`, not enforced.

The value is a triple-quoted string spanning lines, with a **backslash line
continuation** at the end of several lines:

```python
_CLASSIFY_SYSTEM = """You are a clinical triage assistant. Given a patient's \
description of their symptoms, identify the single most likely condition.
```

A trailing `\` inside a string literal swallows the newline that follows it. So
the source can wrap at 88 columns while the resulting string keeps its own
deliberate line structure — the blank lines and the `\n` before `{labels}` are
real, the wrapping in the source is not.

`{labels}` is not special to Python here; it is an ordinary pair of braces that
`str.format` will later look for.

### `.format(...)` (line 72)

```python
system = _CLASSIFY_SYSTEM.format(labels="\n".join(f"- {label}" for label in labels))
```

`str.format` scans the string it is called on for `{name}` placeholders and
replaces them with the matching keyword argument. It is the older sibling of
f-strings; it is the right tool here precisely *because* the template is defined
far away from the values, at module level, where an f-string would be evaluated
too early.

Nested inside is a **generator expression** — `f"- {label}" for label in labels`
— producing values lazily rather than building a list, passed straight to
`join`.

And `f"- {label}"` is an **f-string**: a literal prefixed with `f` whose braces
are evaluated *immediately*, at the point the line runs. This is the exact
opposite of `_CLASSIFY_SYSTEM`'s braces, which stay literal until `.format` is
called on them. Two brace syntaxes, two different moments — the distinction is
the whole reason this file can define its prompts at import time.

### Default arguments and `None` as a sentinel (line 65)

```python
def build_classification_messages(symptom_text, labels, answer=None):
```

`answer=None` gives the parameter a **default**, making it optional. `None` is
used as a **sentinel** — a value whose meaning is "nothing was supplied",
distinct from any real value the caller might pass.

A note on defaults generally: a *mutable* default such as `labels=[]` would be a
well-known Python trap, because the default object is created once at definition
time and shared by every call. `None` is immutable, so this file sidesteps it.

### Building the list (lines 73-81)

```python
messages = [
    {"role": "system", "content": system},
    {"role": "user", "content": f"{CLASSIFY_TAG}\n{symptom_text}"},
]
```

A **list literal** containing two **dict literals**. Dicts are key-value maps and
preserve insertion order (guaranteed since Python 3.7). Using dicts with `role`
and `content` keys rather than tuples means the consumer reads
`message["role"]`, which cannot be silently transposed.

`.append(...)` mutates the list in place and returns `None`, which is why its
result is never assigned.

### `if answer is not None:` (line 77)

**`is` versus `==`.** `is` asks whether two names refer to the very same object;
`==` asks whether two values are equal. For `None` — of which exactly one
instance exists — `is` is the correct and conventional test.

The choice of `is not None` over a plain `if answer:` is deliberate and load
bearing. **Truthiness** means an empty string is **falsy**, so `if answer:`
would treat `answer=""` as "no answer given". `is not None` distinguishes "no
answer" from "an answer that happens to be empty". §5 exercise 1 measures the
difference.

### `build_summarization_messages` (line 84)

Structurally identical to its sibling; no new constructs. Its docstring uses a
**cross-reference** instead of repeating the rationale: "See
build_classification_messages for why one function serves both training and
inference."

## 3. What the data looks like

Pass 2: the same walk through a **data** lens, assuming §2 has been read.

This file's data is text and small records — no arrays, no numbers, no tensors.
So the useful description is **key sets, element types, and the invariants each
structure holds**, not shapes or ranges.

### How the values below were obtained

**Observed.** Every value in this section was printed by a probe script run with
`.venv/bin/python` (Python 3.11.15) against this repository, importing
`src.prompts` directly. The module has no imports and no side effects at import
time, so nothing needed isolating. Example inputs are the ones the tests already
use — **Cited**, `tests/test_prompts.py`: `LABELS = ["allergy", "malaria",
"psoriasis"]` and the symptom text `"itchy red rash"`. The probe wrote nothing
into the project tree.

### The constants

| Name | Type | Observed value | Invariant |
|---|---|---|---|
| `CLASSIFY_TAG` | `str` | `'[TASK: CLASSIFY]'` | Must differ from `SUMMARIZE_TAG` (**Cited**: `test_task_tags_are_distinct`). |
| `SUMMARIZE_TAG` | `str` | `'[TASK: SUMMARIZE]'` | As above. |
| `REQUIRED_SECTIONS` | `list[str]`, length 4 | `['Symptoms', 'Diagnosis', 'History of Patient', 'Plan of Action']` | Every element must appear in the summarization system prompt (**Cited**: `test_summarization_system_prompt_names_all_required_sections`). |
| `_CLASSIFY_SYSTEM` | `str`, 264 chars | contains exactly one placeholder, `{labels}` (**Observed**) | — |
| `_SUMMARIZE_SYSTEM` | `str`, 257 chars | contains exactly one placeholder, `{sections}` (**Observed**) | — |

### `normalize_label` — input and output

Takes a `str`, returns a `str`. **Observed** behaviour across a range of inputs:

| Input | Output | What it demonstrates |
|---|---|---|
| `'  Drug Reaction  '` | `'drug reaction'` | trimming and lowercasing |
| `'urinary\n tract  infection'` | `'urinary tract infection'` | any whitespace run collapses to one space |
| `'Malaria.'` | `'malaria'` | trailing punctuation removed |
| `'"malaria"'` | `'malaria'` | punctuation stripped from *both* ends |
| `"Crohn's Disease."` | `"crohn's disease"` | the internal apostrophe **survives** — `.strip` only touches the ends |
| `'ALLERGY!!!'` | `'allergy'` | repeated punctuation, all removed |
| `'type 2 diabetes'` | `'type 2 diabetes'` | digits and internal spaces untouched |
| `'...'` | `''` | **degenerate**: an all-punctuation input normalizes to the empty string |
| `''` | `''` | empty in, empty out — no exception |

The last two rows are worth noting: the function has no failure mode, but it can
produce an empty string, and nothing downstream in this file checks for that.

### The message list — the file's main product

Both builders return the same shape: a `list[dict]` where every dict has
**exactly two keys**, `role` and `content`, both `str` (**Observed**:
`sorted(msg.keys()) == ['content', 'role']`).

| Property | Inference call | Training call |
|---|---|---|
| Length | 2 (**Observed**) | 3 (**Observed**) |
| Roles, in order | `['system', 'user']` | `['system', 'user', 'assistant']` |
| Trigger | `answer` omitted | `answer=` supplied |

The invariant that the rest of the pipeline depends on: **the assistant turn,
when present, is last**. `docs/explainers/tokenization.md` §2 shows
`build_training_example` slicing it off with `messages[:-1]` to find the prompt;
that slice is only correct because of the ordering established here.

**Observed** classification messages for the test fixture:

```
[
  {"role": "system",
   "content": "You are a clinical triage assistant. Given a patient's description of\n"
              "their symptoms, identify the single most likely condition.\n"
              "\n"
              "Respond with only the condition name, exactly as written in the list\n"
              "below. Do not explain your reasoning.\n"
              "\n"
              "Valid conditions:\n"
              "- allergy\n- malaria\n- psoriasis"},
  {"role": "user",
   "content": "[TASK: CLASSIFY]\nitchy red rash"},
  {"role": "assistant",
   "content": "psoriasis"}
]
```

**Observed** summarization system content, rendered from `REQUIRED_SECTIONS`:

```
You are a clinical documentation assistant. Summarize the following
doctor-patient conversation into a structured clinical note.

Use exactly these four sections, each on its own line:
Symptoms:
Diagnosis:
History of Patient:
Plan of Action:

Write "N/A" for any section the conversation does not cover.
```

Note the trailing colon on each section name: it comes from `f"{section}:"` in
the builder, not from `REQUIRED_SECTIONS` itself, whose entries have no colon
(**Observed**, see the constants table). Anything that compares model output
against `REQUIRED_SECTIONS` must account for that difference.

**Observed**: two calls with identical arguments return equal but **distinct**
objects (`a == b` is `True`, `a is b` is `False`, and even `a[0] is b[0]` is
`False`). Each call builds fresh dicts, so a caller mutating one result cannot
affect another.

### Data flow

The path through this file is linear — build a system string, build a user
string, optionally append one more dict — with a single optional step and no
loops or back-edges. Prose covers it, so no flow diagram is included here. (The
one branch, on `answer`, is the table above.)

### Borrowed-API ledger

"Theirs" means parameter names defined outside this repository; "ours" means the
values supplied for them. Every entry here is a Python builtin — this file has no
third-party dependencies at all (**Observed**, §0).

| Call | Package | Returns | Params (theirs) | Params (ours) | Version risk |
|---|---|---|---|---|---|
| `str.format(**kwargs)` | stdlib | `str` (**Observed**: 285-char classification system prompt for 3 labels) | the placeholder names in the template | `labels=`, `sections=` | None. **Inference**: `str.format` is frozen public API; it has been stable since Python 2.6. |
| `str.strip(chars)` | stdlib | `str` | `chars`, a *set* of characters | `'.,;:!?"\''` | None. **Inference**: same reasoning. |
| `str.split()` / `str.join()` | stdlib | `list[str]` / `str` | separator (omitted, meaning any whitespace run) | — | None. |

## 4. Why it is built this way

Pass 3: the same walk through a **design** lens. Recorded rationale is cited;
my own reading is marked.

### The concepts actually in play

- **Prompts as a contract.** **Cited**, the module docstring: "That comparison is
  only meaningful if both models receive exactly the same prompt. Defining
  prompts in one place, imported by both data_prep.py and evaluate.py, makes
  divergence impossible rather than merely unlikely." Anchored in the fact that
  both builders are the *only* definitions of prompt text — there is no string
  literal resembling a prompt anywhere else in `src/`.
- **One function for training and inference.** Anchored at
  `build_classification_messages` — `if answer is not None:` followed by
  `messages.append(...)`. **Cited**, its docstring: "Using one function for both
  is what keeps train-time and test-time prompts identical."
- **The label list inside the prompt.** Anchored at
  `system = _CLASSIFY_SYSTEM.format(labels="\n".join(f"- {label}" for label in labels))`.
- **Task tags as a routing signal.** Anchored at
  `{"role": "user", "content": f"{CLASSIFY_TAG}\n{symptom_text}"}`.
- **Charitable normalization.** Anchored at `normalize_label` —
  `return cleaned.strip('.,;:!?"\'')`.

### Decisions, and what was rejected

**Putting the 22 labels in the prompt rather than letting the model memorize
them.** The most thoroughly argued decision in the file. **Cited**, the module
docstring, which names two reasons: *fairness* — "The baseline is an untuned
model; asked to pick from an unstated label set it would fail for the wrong
reason, and the comparison would be rigged in our favour" — and *sharpening the
claim* — "the tuned model is not being credited with memorising a label space —
it is being credited with reliably mapping symptom language onto it". **Cited**
again in `docs/superpowers/specs/2026-08-17-healthcare-lora-sft-design.md`: "The
label list appears in both training and evaluation prompts, identically for base
and tuned models."

**Passing `labels` as a parameter rather than hard-coding them.** No recorded
rationale. **Inference**: the spec says the label space is derived from the
training split only and written to `data/labels.json` — "Deriving them from the
test split would leak" — so the list is *data*, discovered at runtime, and
cannot be a literal in this file. That also makes the module trivially testable
with a three-label list, which is what the tests do.

**A bare label as the training target.** **Cited**, the inline comment: "The
target is the bare label. Any preamble ('The condition is...') would have to be
stripped at eval time, adding a parsing failure mode." Note the reasoning is
about the *evaluator*, not the model — a shorter target is chosen to keep the
scoring code simple and unambiguous.

**Normalizing rather than requiring exact matches.** **Cited**, the
`normalize_label` docstring: applied "to gold labels in data_prep and to model
predictions in metrics, so the baseline is not penalised for cosmetic
differences like a trailing full stop. Being charitable to the baseline is what
makes the reported gain credible." **Cited**, `tests/test_prompts.py`:
"Base-model outputs routinely arrive as 'Malaria.' — without this the baseline
gets scored down for punctuation rather than for being wrong." The design goal
is a *credible* comparison, not a flattering one.

**Why `normalize_label` lives here at all.** It is not prompt construction, and
its placement looks arbitrary until the docstring explains it: "It lives in this
module because the label vocabulary is part of the prompt contract, and because
metrics.py needs it without pulling in `datasets`." **Cited**. So the placement
is a dependency decision — the alternative, putting it in a data module, would
drag a heavy import into the scoring path.

**Task tags.** **Cited**, `docs/superpowers/specs/2026-08-17-healthcare-lora-sft-design.md`:
"**Task tags** are cheap insurance." **Inference**: since a single adapter is
trained on both jobs and the two system prompts already differ, the tag is
redundant in the common case — it is there for the case where the model attends
weakly to a long system prompt but reliably to a short marker near the user
text.

**`is not None` rather than truthiness.** No recorded rationale. **Inference**:
an empty assistant turn is a legitimate thing to want to represent — a
summarization example whose reference note is empty, say — and conflating it
with "no answer" would silently turn a training example into an inference
prompt. §5 exercise 1 shows the mechanism.

### Where the bugs hide

**What this code defends against:**

- Prompt drift between training and evaluation, by construction: one function,
  two call sites.
- Cosmetic scoring failures, via `normalize_label`.
- Ambiguous parsing of model output, by making the target a bare label.
- Disclaimer text leaking into a training target — **Cited**,
  `test_no_disclaimer_leaks_into_prompts`: "A disclaimer inside a training
  target would be learned as output and would wreck ROUGE."

**What it does not defend against** (**Inference** throughout — gaps I found by
looking, not documented limitations):

- `REQUIRED_SECTIONS` is a **mutable module-level list**. Any importer can append
  to it and change every subsequent prompt, process-wide. §5 exercise 2.
- `labels=[]` produces a system prompt ending "Valid conditions:" with nothing
  after it — **Observed** in §5 exercise 5. No validation, no warning.
- `normalize_label` can return `''` (§3), and nothing checks. An empty
  prediction would then match an empty gold label.
- The comment on `REQUIRED_SECTIONS` says "keep the two in sync" with
  `metrics.py`. That is a **manual** invariant with no test behind it, and
  `metrics.py` does not exist yet — the highest-risk line in the file, precisely
  because the coupling is documented only in a comment.
- Nothing checks that `answer`, when supplied, is one of `labels`.

### How the tests pin it down

**Cited**, `tests/test_prompts.py`, whose own docstring states the stakes: "a
silent drift between the training prompt and the evaluation prompt would
invalidate every number the project reports, while still 'working'."

| Test | What would slip through without it |
|---|---|
| `test_normalize_lowercases_and_strips`, `..._collapses_internal_whitespace`, `..._strips_surrounding_punctuation` | Each of the three normalization behaviours independently. |
| `test_classification_inference_messages_have_no_assistant_turn` | An assistant turn leaking into an inference prompt, which would hand the model its own answer. |
| `test_classification_training_messages_end_with_answer` | The ordering invariant that `build_training_example` slices on. |
| `test_classification_system_prompt_lists_every_label` | A dropped or truncated label list. |
| `test_..._user_turn_carries_task_tag_and_text` (both tasks) | A lost task tag, or lost user content. |
| `test_summarization_system_prompt_names_all_required_sections` | Drift between `REQUIRED_SECTIONS` and the rendered prompt. |
| `test_task_tags_are_distinct` | The two tags being made equal in a careless edit. |
| `test_no_disclaimer_leaks_into_prompts` | A safety disclaimer being learned as output. Parameterized over both builders. |

**Inference**: the suite is strong on structure and weak on degenerate inputs —
nothing covers empty `labels`, an all-punctuation label, or mutation of
`REQUIRED_SECTIONS`, all three of which §5 shows to be silent.

## 5. Break it

Change one thing, predict, then look. Every answer was obtained by copying
`src/prompts.py` to a scratchpad, editing the copy, and running it.

1. Change `if answer is not None:` to `if answer:` in both builders, then call
   with `answer=""`. What differs?
2. After importing the module, run `REQUIRED_SECTIONS.append("Follow-up")`, then
   build a summarization prompt. Does the prompt change?
3. In `normalize_label`, reorder the two steps — strip punctuation first, then
   trim and collapse whitespace. Do any results change?
4. Pass a label containing a format placeholder, e.g. `"{sections}"`. Does
   `.format` recurse into it and raise?
5. Pass `labels=[]`. What does the system prompt look like?

<details><summary>Answers</summary>

1. **The example silently changes category.** Observed: with the original code,
   `answer=""` yields roles `['system', 'user', 'assistant']` with an assistant
   turn whose content is `''`. With `if answer:`, the same call yields
   `['system', 'user']` — the assistant turn vanishes. A non-empty answer behaves
   identically under both.

   So the modification converts a training example into an inference prompt
   whenever the reference answer happens to be empty, without raising anything.
   Downstream, `build_training_example` would then mask the whole sequence and
   train on nothing. This is the practical difference between `is not None` and
   truthiness (§2).

2. **Yes — every subsequent prompt changes, process-wide.** Observed: the system
   content before and after the append differ, and `REQUIRED_SECTIONS` becomes
   `['Symptoms', 'Diagnosis', 'History of Patient', 'Plan of Action',
   'Follow-up']`. The prompt gains a fifth section line while the prose above it
   still reads "Use exactly these four sections".

   The list is read at *call* time, not captured at import time, so there is no
   snapshot protecting earlier callers. `UPPER_CASE` is a request, not a lock.

3. **Yes, and it breaks exactly the case the function exists for.** Observed
   differences:

   | Input | Original | Reordered |
   |---|---|---|
   | `'Malaria.'` | `'malaria'` | `'malaria'` |
   | `'  Malaria.  '` | `'malaria'` | `'malaria.'` |
   | `'  "Malaria."  '` | `'malaria'` | `'"malaria."'` |

   Stripping punctuation first leaves the trailing space in place, so the
   punctuation is no longer *at* the end and survives. Inputs without surrounding
   whitespace are unaffected — which is what makes this dangerous: the unit tests
   as written all pass strings that would still be normalized correctly, and the
   real inputs (**Cited**, model output like `"Malaria. "`) are the ones that
   would break.

4. **No exception; the placeholder passes through untouched.** Observed: with
   labels `["allergy", "{sections}", "psoriasis"]`, the system prompt ends
   `'Valid conditions:\n- allergy\n- {sections}\n- psoriasis'`. A lone unbalanced
   brace, `"a{b"`, also survives intact.

   `str.format` processes only the template it is called on. The *substituted
   values* are inserted as-is and never rescanned, so braces arriving through
   `labels` are inert. (Had the code built the prompt with an f-string over
   untrusted text, or called `.format` twice, this would not hold.)

5. **A prompt that asks the model to choose from nothing.** Observed: the system
   content ends `'Do not explain your reasoning.\n\nValid conditions:\n'` — the
   header is there, the list is empty. Roles are still `['system', 'user']`; no
   error, no warning.

   `"\n".join(...)` over an empty sequence returns `''`, which `.format`
   substitutes happily. Every layer behaves correctly and the result is useless —
   the kind of failure that only shows up as a mysteriously terrible evaluation
   score.

</details>

## 6. Glossary

Terms shared with `docs/explainers/tokenization.md` carry the same definitions
there. Where the same word means something different in the two files, that is
called out explicitly below.

- **Baseline** — the untuned base model, evaluated on the same test set with the
  same prompts, as the comparison point for the fine-tuned one.
- **Constant (Python)** — a module-level `UPPER_CASE` name. A naming convention
  only; Python has no `const` and does not prevent reassignment or mutation.
- **f-string** — a literal prefixed with `f` whose `{...}` are evaluated
  immediately, where the line appears. Contrast `str.format`.
- **Gold label** — the correct answer recorded in the dataset, against which a
  prediction is scored.
- **Immutable** — cannot be changed in place. Python strings are; lists are not.
- **Invalid-label rate** — the fraction of predictions matching none of the valid
  labels. A headline metric for this project, per the design spec.
- **Label (classification)** — a condition name such as `"malaria"`; one member
  of the label space. **Not** the same as `labels` in
  `docs/explainers/tokenization.md`, which there means the list of token ids the
  loss is computed against. Same word, two unrelated meanings, one per file.
- **Label space** — the fixed set of valid labels; 22 in this project, derived
  from the training split only.
- **Message** — one `{"role": ..., "content": ...}` dict. A prompt is a list of
  them.
- **Mutable default trap** — giving a parameter a mutable default (`x=[]`), which
  is created once at definition time and shared across calls. Avoided here by
  defaulting to `None`.
- **Normalization (text)** — reducing a string to a canonical form so that
  cosmetic variants compare equal.
- **Prompt contract** — the guarantee that training and evaluation use
  byte-identical prompt construction. The reason this module exists.
- **ROUGE** — the overlap-based score used to evaluate the generated summaries.
- **Role** — `system`, `user`, or `assistant`; which participant a message
  belongs to.
- **Sentinel** — a value chosen to mean "nothing here", distinct from a real
  value. `None` for an unsupplied answer. (Same meaning as in the tokenization
  explainer.)
- **SFT (supervised fine-tuning)** — next-token prediction on (prompt, answer)
  pairs with the loss restricted to the answer. (Same meaning as in the
  tokenization explainer.)
- **`str.format`** — placeholder substitution evaluated when `.format` is called,
  not when the template literal is written. What lets a prompt template be
  defined at module level.
- **System prompt** — the instruction message, role `system`, that frames the
  task before the user's content.
- **Task tag** — a short marker (`[TASK: CLASSIFY]`) at the head of the user turn
  identifying which job this example is.
- **Truthiness** — an object's implicit boolean value; empty containers, `0`, and
  `None` are falsy. (Same meaning as in the tokenization explainer.)

## 7. My notes

_Yours. Preserved verbatim when this document is regenerated._
