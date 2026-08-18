# SDD ledger — plan: docs/superpowers/plans/2026-08-18-explain-code-skill.md

Spec: docs/superpowers/specs/2026-08-18-explain-code-skill-design.md (read, binding authority)
Skills repo (Tasks 1-8): ~/Developer/claude-skills (created by Task 1)
Host repo (Tasks 9-10): ~/Developer/healthcare-sft @ feat/healthcare-lora-sft

## Pre-flight scan

| # | Tasks / scope | Produces vs consumes | Finding |
|---|---|---|---|
| 1 | T2 -> T3 | T2 produces `split_sections`, `SECTIONS`, `HEADING`; T3 consumes `split_sections` | agree |
| 2 | T2 -> T4 | T2 produces the check_* trio; T4's `validate()` calls all three | agree |
| 3 | T3 -> T4 | T3 produces `source_symbols`, `check_coverage`; T4's `validate()` calls `check_coverage` when suffix == .py | agree |
| 4 | T2 -> T5 | T2 produces `SECTIONS` + `HEADING`; T5's test compares template headings to `SECTIONS` | agree; template titles verified char-identical to SECTIONS |
| 5 | T4 -> T9/T10 | T4 produces CLI `<md> [--source] [--notes]`; T9/T10 invoke exactly that | agree |
| 6 | T2 test file -> T3/T4/T5 | T2 defines `SKILLS`, `VALIDATOR`, `_load`, `_doc`, `GOOD_HEADER`; later tasks append tests using them | agree |
| 7 | T1 -> T5 | T1 creates `references/.gitkeep`; T5 deletes it | no conflict: directory still holds document-template.md |
| 8 | T1 -> T2/T8 | T1 creates `scripts/.gitkeep`; T2 fills the dir; T8 deletes the placeholder | redundant from T2 onward, harmless — see Ruling 3 |
| 9 | T1 -> T8 | T1 writes stub SKILL.md; T8 fully replaces it | agree, intended |
| 10 | T2/T3/T4/T5 self-consistency | claimed cumulative test counts 9 / 15 / 27 / 28 | agree; counted by hand against the listed test functions |
| 11 | T6/T7 self-consistency | verification greps for stray `## N.` headings in the reference files | agree; neither file uses numbered headings |
| 12 | T4 self-consistency | appends `import subprocess` / `import sys` mid-file | conflicts with review rubric — see Ruling 1 |
| 13 | T9/T10 self-consistency | steps invoke `/explain-code`, a skill installed during this same session | BLOCKING — see Ruling 2 |
| 14 | T3 self-consistency | `check_coverage` matches bare symbol names as substrings | loose, accepted — see Ruling 4 |
| 15 | T1 branch safety | new repo, `git init`, commits land on its default branch | acceptable — see Ruling 5 |

## Rulings

Ruling 1: T4's implementer may hoist `import subprocess` / `import sys` to the
top of the test file rather than appending them mid-file as the plan's diff
implies. — The plan appends them only because it is written as sequential
patches; top-of-file imports are behaviour-identical and match the rubric a
reviewer will apply. — Cost if wrong: nil; purely cosmetic placement.

Ruling 2: Tasks 9-10 will follow the procedure by READING
`~/.claude/skills/explain-code/SKILL.md` and its references by path, rather
than invoking `/explain-code`. — Claude Code loads skills at session start, so
a skill installed by Task 1 of this same session is not invocable by a
subagent dispatched from it; the plan's "start a new session first" is not
available to a subagent. Reading the file exercises the procedure, the
template, and the validator — everything the acceptance criteria actually
test. — Cost if wrong: skill DISCOVERY goes unverified (frontmatter, name
matching). Mitigated by T1 step 6 checking the frontmatter, and I will flag it
for the user to confirm in a fresh session.

Ruling 3: `scripts/.gitkeep` stays tracked until Task 8 deletes it, as the plan
says, even though Task 2 makes it redundant. — Not worth a plan deviation. —
Cost if wrong: one stray empty file for six tasks.

Ruling 4: `check_coverage`'s substring matching of bare symbol names is
accepted for v1. — Its failure mode is a false PASS (symbol `helper` matched
inside the word `helpers`), never a false failure. As a completeness backstop
whose job is catching omissions, an occasional missed omission is tolerable;
spurious failures would not be. — Cost if wrong: the validator misses a symbol
that was listed but never really explained.

Ruling 5: Task 1 commits to the new repo's default branch. — It is a brand-new
private repo with no history and no shared branch; there is nothing to isolate
from, and an initial commit has to land somewhere. Tasks 9-10 commit to
healthcare-sft's existing feature branch, not main. — Cost if wrong: nil.

## Progress

Task 1: complete (commit 1d458fa, review clean — spec met on all 10 requirements, no scope creep, quality approved)
  note: reviewer observed README's `pytest tests/ -v` exits 5 on the empty tests/ dir; resolves at Task 2, not a finding.
Task 2: implemented (commit 36eb107, 9 tests passing). Review returned spec clean, quality NOT approved: 2 Important, 1 Minor.

Ruling 6: The reviewer's two Important findings are upheld against the plan text
that mandated the exact code and the exact test list. — The spec states the
validator exists to automate "the subset that fails *silently*". A
`split_sections` that silently misattributes a section body when the document
contains a fenced code block quoting a heading-like line contradicts that
purpose on the tool's own core primitive, and the reviewer demonstrated it
rather than hypothesising it. Generated explainers will always contain fenced
code blocks. Likewise, two `check_header` branches survive deletion with the
suite green, so they are not actually pinned. The spec is the binding
authority and it backs the reviewer over the plan's draft code. — Cost if
wrong: a slightly larger Task 2 diff than planned; the function signatures are
unchanged, so Tasks 3-5 are unaffected.
Task 2: minor (deferred): `check_header` uses `.endswith("explainer")` with no
  word boundary; would false-pass a title ending "reexplainer". Real but
  implausible for source filenames. Not entering the fix loop.
Task 2: fix round 1/5 (2 addressed, 0 open; commit 36eb107..91fd330)
  Re-review independently reproduced the pre-fix corruption at 36eb107, confirmed
  it is gone at 91fd330, and mutation-killed both new check_header tests.
Task 2: minor (deferred): an UNCLOSED trailing fence makes `_fenced_ranges` treat
  everything after it as fenced to EOF, swallowing later real headings. New
  failure mode introduced by the fix, but it fails LOUD (missing-section errors
  for every later section) rather than silent, so it is not the bug class the
  validator exists to catch. Triage at final review.
Task 2: complete (commits 1d458fa..91fd330, review clean)

Ruling 7: The plan's expected cumulative test counts are superseded. Task 2's
fix added 5 tests (9 -> 14), so the plan's "15 / 27 / 28" for Tasks 3 / 4 / 5
are stale by five. Corrected expectations: Task 3 -> 20, Task 4 -> 32,
Task 5 -> 33. Task briefs still state the old numbers; each dispatch carries
the correction. — The counts are a verification aid, not a requirement; the
requirement is that every listed test exists and passes. — Cost if wrong: an
implementer or reviewer chases a phantom count mismatch.
Task 3: implemented (commit 29a035d, 20 tests). Spec clean; quality approved with 2 Important.

Ruling 8: Both of Task 3's Important findings are upheld against the plan text
that mandated the exact code. — The spec's Completeness mechanism exists to turn
"explain everything" into "a checkable postcondition", and the validator's stated
job is the subset that fails SILENTLY. (a) `source_symbols` dropping
tuple-unpacked module-level assignments means such a symbol never enters the
postcondition at all and the document reports full coverage while omitting it —
a silent hole in the tool's namesake guarantee. (b) `check_coverage` raising
`UnicodeDecodeError` on a non-UTF-8 source breaks the uniform "always returns a
list of failure strings" contract that Task 4's `validate()` concatenates, so
one stray Latin-1 byte crashes the whole run instead of reporting a failure.
Neither was considered by the plan rather than deliberately chosen by it. — Cost
if wrong: a slightly larger Task 3 diff; no signature changes, so Tasks 4-5 are
unaffected.
  Distinguished from Ruling 4: that ruling accepted loose SUBSTRING matching
  inside check_coverage, whose failure direction is a false pass on a symbol
  that IS in the inventory. Finding (a) is a symbol never reaching the
  inventory requirement at all. Different mechanism, not covered by Ruling 4.
Task 3: fix round 1/5 (2 addressed, 0 open; commits 29a035d..f63a909)
Task 3: minor (deferred): `_record_constant_targets` does not handle `ast.Starred`
  in an unpack target (`A, *B = 1, 2, 3` records only A). Same silent-drop class
  as the finding just fixed, but strictly better than pre-fix and not asked for.
Task 3: minor (deferred): `source_symbols` re-parses the source on every
  `check_coverage` call; fine at one-file-per-invocation scale.
Task 3: complete (commits 91fd330..f63a909, review clean)

Ruling 7 amended: Task 3's fix added 3 more tests (20 -> 23). Corrected running
expectations: Task 4 -> 35, Task 5 -> 36. Same rationale as Ruling 7.
Task 4: implemented (commit be8c8c5, 35 tests). Spec clean; quality approved with 1 Important + 3 Minor.
  Note: two reviewer dispatches were killed mid-run by machine sleep (API error).
  Repo verified clean at be8c8c5 after each; third attempt resumed and completed.

Ruling 9: `check_exercises`'s Important finding is upheld and treated as a
BLOCKER, not the fast-follow the reviewer suggested. — The reviewer constructed
the brief's own mandated shape (numbered answers matching numbered exercises)
and got failures in both directions: 3 exercises + 3 numbered answers reports
"found 6" and REJECTS a correctly-formed document. Task 9 step 3 requires the
validator to exit 0 on a generated explainer, and Task 5's template mandates
numbered answers. Left alone, this fails acceptance on a correct document. —
Cost if wrong: a slightly larger Task 4 diff.

Ruling 10: `extract_notes` is elevated from the reviewer's Minor to part of this
fix round. — It inherits `split_sections`, so a reader whose own section-7 notes
contain an unfenced line shaped like `## 3. things I still don't get` has
everything after it silently dropped when the skill regenerates over the file.
That shape is plausible in freeform notes, and the spec is explicit that
destroying a reader's annotations "would make the reader reluctant to annotate
at all" — which makes this load-bearing rather than cosmetic. The fix does NOT
require touching frozen Task 2/3 code: section 7 is by definition the last
section, so reading from its heading to end-of-file is both simpler and immune
to whatever the reader writes. — Cost if wrong: `extract_notes` diverges from
`split_sections` in how it finds section 7; acceptable, since it is the only
consumer that must survive arbitrary user prose.
Task 4: minor (deferred): `LINE_REFERENCE` matches only literal `line \d+`;
  "L119", "row 119", "at 119" slip through unflagged.
Task 4: minor (deferred): no test covers `validate()`'s `.py`-suffix gate; the
  gate is correct (verified live via CLI) but mutates freely without failing.
Task 4: fix round 1/5 (2 addressed, 0 open; commits be8c8c5..65850bb)
Task 4: complete (commits f63a909..65850bb, review clean)
  Validator is finished: 41 tests, stdlib-only, CLI verified end to end.
Running expectation amended again: Task 5 -> 42.
Task 5: complete (commits 65850bb..d347652, review clean, 42 tests)
  Reviewer hand-authored a document by following the template and ran it through
  the real validator: exit 0. Template and validator proven to agree end to end.
  Pinning test mutation-verified (retitled a section -> test failed).
Task 5: minor (deferred): §5's illustrative format block shows only two numbered
  items ("1. ... / 2. ..."); a writer copying it literally instead of reading it
  as a sketch would trip the 3-5 bound. Prose immediately above says "Three to
  five", so a careful writer is fine.
Tasks 6+7: BATCHED into one dispatch — both are pure reference-file authoring of
  identical shape (write markdown, grep for stray numbered headings), no code, no
  TDD, and a reviewer could not meaningfully approve one while rejecting the
  other. Reviewed as one unit.
Tasks 6+7: implemented (commits 2c91fb8, 3b906e8; 42 tests). Spec clean, byte-exact
  to briefs. Quality approved with 2 Important cross-file contradictions.

Ruling 11 (Finding A — Pass 1 ordering): The TEMPLATE wins; `python-constructs.md`
is amended. — Template §2 says walk the file top to bottom; the checklist says
"work down this list", which would have a writer group by category and jump
around the source. Two writers would produce materially different §2 sections.
The spec settles it: Pass 1 is "Top-to-bottom walk. Every Python construct named
and explained on FIRST APPEARANCE." First appearance is a property of source
order, so the file order is authoritative and the checklist is a coverage aid
consulted during that walk, not an ordering to follow. — Cost if wrong: §2 reads
in source order rather than grouped by construct family.

Ruling 12 (Finding B — is the mermaid diagram mandatory?): `diagram-recipes.md`
wins; the TEMPLATE is amended. — Template §3 lists "One mermaid flowchart of data
movement through the file" with no condition, while diagram-recipes.md says a
diagram is included only when it carries information prose cannot. The spec is
explicit and matches the recipes file: "A diagram is included only when it
carries information the prose cannot. No decorative diagrams." An unconditional
mandate would have Task 9 emit a decorative flowchart for a linear data path. —
Cost if wrong: some explainers omit a flowchart a reader might have liked.
  Editing document-template.md touches only §3's body, not any heading, so the
  Task 5 pinning test is unaffected. Implementer must confirm 42 still pass.
Tasks 6+7: fix round 1/5 (2 addressed, 0 open; commits 3b906e8..c4ee049)
Tasks 6+7: minor (deferred): the template's paraphrase of the diagram test names
  only diagram-recipes case 2 (branching/looping) plus a pointer, not case 3
  (relationships among >2 things). Not a contradiction — the bullet defers
  explicitly to diagram-recipes.md for the full test.
Tasks 6+7: complete (commits d347652..c4ee049, review clean, 42 tests)
  Re-reviewer read all three reference files together and confirmed two writers
  would now produce structurally consistent §2 and §3.
Task 8: implemented (commit 79f4c6a, 42 tests). Spec clean; quality approved with
  1 Important + 1 Minor. Reviewer confirmed the implementer's two judgement calls
  were both correct (installation note needed no correction — git history shows the
  stale text never existed in this file; and making the two post-plan decisions
  explicit in step 7 was the more defensible choice).

Ruling 13: The `python` vs `python3` finding is upheld and blocking. — SKILL.md
steps 2 and 10 invoke the validator as `python <path>`, but bare `python` is not
on this machine's PATH (only `/opt/homebrew/bin/python3`). The reviewer confirmed
this empirically. SKILL.md's own step 5 already knows the problem, saying "prefer
`.venv/bin/python`, fall back to `python3`", but that fallback was never applied
to the two validator invocations. Task 9 step 3 runs exactly this command. Left
alone the procedure fails at its first and last steps. Inherited verbatim from
the brief, so a ruling rather than an implementer error. — Cost if wrong: nil,
`python3` is strictly more available than `python`.

Ruling 14: The caller-search grep's hardcoded `--include="*.py"` is pulled INTO
this fix round despite the reviewer classing it Minor. — Task 10 runs the skill
against a JavaScript file and reads §0 for correctness; a Python-only grep
silently finds nothing and yields a false "no callers" orientation rather than
erroring. It is one line, in the file already being edited, and Task 10 exercises
exactly that path — cheaper to fix now than to have Task 10 fail on it. — Cost if
wrong: the search widens and may surface incidental textual matches; §0 asks for
judgement about callers anyway, so a slightly wider net is the safer error.
Task 8: fix round 1/5 (2 addressed, 0 open; commits 79f4c6a..3b9c1f1)
  Re-reviewer ran both validator commands as written and confirmed they execute;
  also reproduced the pre-fix `command not found: python` (exit 127).
Task 8: minor (deferred): Claude Code's grep wrapper omits the leading `./` on
  top-level paths, so the step-4 exclusion regex `/(...)/` misses noise
  directories sitting AT the search root. Verified correct under a real grep
  binary, and masked in both repos in play because their .gitignore covers it.
  Environmental quirk of this harness, not something SKILL.md can code around.
INCIDENT: the Task 8 re-reviewer ran `git checkout -- src/tokenization.py` in
  healthcare-sft, discarding the user's uncommitted trailing-newline edit. Content
  lost is one blank line. Reported to the user rather than silently restored,
  since it is theirs to decide.
Follow-on: document-template.md line 22 still hardcodes `--include="*.py"` for the
  §0 caller search, now inconsistent with SKILL.md step 4 after Ruling 14. Task 10
  reads §0 on a JavaScript file, so dispatching a one-line fix before Task 9.
Tasks 6+7: follow-on fix (commit eaedb95) — template caller search aligned with
  SKILL.md step 4. 42 tests, 8 headings unchanged.
Task 9 pre-flight: Qwen/Qwen2.5-1.5B-Instruct IS in the local HF cache, so the
  probe can run offline and §3 should carry OBSERVED values rather than
  "Not executed" labels. This is the stronger acceptance path; if the document
  comes back with unlabelled guesses OR needless "Not executed" labels, that is a
  procedure failure in SKILL.md step 5, not a document nitpick.
Task 9: complete (healthcare-sft 13c5466; claude-skills 640552e — 5 skill defects)
Task 10: complete (healthcare-sft e63848f; claude-skills 5ad2357 — 3 skill defects)
  Both explainers validate at exit 0. All six acceptance questions answered.
  Notes survived regeneration verbatim, exactly once. All four JS generality
  checks pass. 42 skills tests and 21 host tests green.
Ruling 15: the final whole-branch review runs on sonnet rather than the most
capable model, contrary to the skill's Model Selection guidance. — The user hit a
monthly spend limit mid-run and asked to wrap up; sonnet has caught every real
defect this session, including three the implementers had only flagged as
speculative. — Cost if wrong: a subtler cross-cutting issue escapes to the
finishing step, where the user sees it before merging.

## Final whole-branch review

Zero factual errors found in the generated document. Reviewer re-ran the probe
offline and independently reproduced all five "Break it" exercises from live
code; every Observed number matched, every Cited claim traced to its source,
every Inference was genuinely unsupported rather than smuggled in as fact. All
six spec acceptance questions verified independently.

Ruling 16: No fix wave. All three residual findings are Minor and I am parking
them. — (a) "MRO" is used at tokenization.md:409 before the glossary defines it
at :780, a real rule-6 violation, but the adjacent table row reads `__mro__` and
the concept is legible from the printed chain; more importantly, hand-editing a
GENERATED document contradicts the project's own principle that defects belong
in the skill, and a regeneration would discard the edit anyway. (b) The
no-checklist-for-unknown-languages fallback lives in SKILL.md step 7 but not in
document-template.md §2; it worked correctly in the JS run. (c) The spec lists
the inventory postcondition as "universal", but `validate()` only runs coverage
for `.py` sources, so completeness is writer discipline for other languages —
the spec overstates what is mechanically guaranteed. — Cost if wrong: three
small blemishes reach the user, all documented here rather than discarded.

Ruling 17: The skill-discoverability gap (Ruling 2) is handed to the user rather
than chased. — The reviewer noted `explain-code` is absent from its own
available-skills listing, but a subagent inherits the skill list fixed at its
parent session's start, which predates the install; that is the expected
observation and is NOT evidence of a defect. The frontmatter, symlink and file
layout were all verified correct. The only real test is a brand-new interactive
session, which I cannot create. — Cost if wrong: the user finds `/explain-code`
does not appear and needs a frontmatter or path fix.
