---
allowed-tools: Bash(git status:*), Bash(git diff:*), Bash(git log:*), Bash(gh pr:*), Bash(uv run:*), Bash(python:*), Bash(sqlite3:*), Bash(grep:*), Bash(jq:*)
description: Review the branch diff against this repo's measurement invariants, then craft the conventional PR title
model: sonnet
disable-model-invocation: true
---

# I'll review the branch diff against this repo's measurement invariants, then craft the conventional PR title and description

Under this repo's git workflow (see CLAUDE.md), `main` only receives commits in
Conventional Commits form. The conventional format therefore applies to exactly
one artefact — the PR title, which becomes the commit message on `main` — and
that is what this command produces, plus the PR description. It never creates
commits.

## Phase 1 — Review

Review the branch diff (`git diff main...HEAD`) against the checklist below.
Only flag issues that appear **in the diff itself** (changed or added lines).

This checklist is not generic: every item corresponds to a defect that actually
shipped in this repository and was caught late, usually by a reviewer rather
than by a test. Three review passes on milestone 3 found zero wrong result
figures and a dozen wrong statements *about* those figures — so weight the
documentation items as heavily as the code ones.

**Measurement traceability** — the repo's core promise

- A figure published in `README.md`, `docs/eval-*.md` or `JOURNAL.md` that is
  not recomputable from a JSON under `docs/mesures/`, produced by a versioned
  script. Recompute two of them yourself from the raw `par_question` dicts.
- A **derived** figure (a factor, a duration, a count of configurations, a
  percentage) that no longer follows from the JSONs it was computed from. This
  is the single most repeated defect: `5,8×` when the numbers now give `6,0×`.
- An ad hoc measurement published without either its JSON or an explicit
  statement that it is unversioned.
- A latency quoted without its **device**. On this machine the same reranking
  costs ~1.5 s per question on GPU and ~150 s on CPU; a bare number in seconds
  is meaningless, and a ratio quoted to one decimal is false precision — four
  runs of the same probe spanned ×66 to ×106.

**Ablation protocol**

- An ablation that moves **two** parameters at once and reads the result as the
  effect of one of them. If a second parameter had to move, the configuration
  isolating the first must exist in the same grid.
- An adoption criterion substituted after the measurement (the criterion is
  `p_amelioration >= 0.95` on **recall@10**, plus no category losing more than
  0.05). A gain on recall@5 or MRR is worth reporting, never worth adopting on.
- Any use of the frozen `benchmark/test.jsonl` to choose a parameter, or more
  than one execution of it per milestone closing.

**Controls that cannot fail**

- An integrity control, a freshness control or a reproduction control with no
  test that makes it fail. A control nobody has seen fail proves nothing — this
  repo shipped a reproduction control that validated the very bug it was meant
  to catch, because the control demanded agreement with a figure nobody
  re-derived.
- A test whose assertion holds regardless of the implementation: `assert A == B
  or <condition that rescues the failure>`, a spy using `setdefault(...) or ...`
  (which breaks when the observed value is 0), or an assertion on a property
  that the surrounding code guarantees unconditionally.
- **Apply one mutation.** Break the line the new test is supposed to guard and
  confirm the test fails, then restore. State in the review that you did.

**Claims and mechanisms**

- A mechanism asserted ("because", "explained by", "the cause is") with no
  measurement behind it. Three of these shipped in milestone 3 and all three
  were wrong while their conclusions were right.
- A statement contradicting another passage of the same document, or the same
  claim corrected in one of `README.md` / `docs/eval-*.md` / `JOURNAL.md` but
  not the other two. Grep the other two for the old figure.
- A figure cited without the caveat that qualifies it, when the report attaches
  one (sample size, attainability versus recall, unmeasured cost).

**Corpus, benchmark and secrets**

- A rewriter, generator or judge that receives the expected citations, the
  corpus, or the retrieval results — the model must see only the question.
- A benchmark question whose gold citation cannot be established from the
  corpus, or a question derived from the very document that answers it
  (rescrits: held out of the measurement index, or the measurement is
  circular).
- An index rebuild in a milestone that declared none, or a change to
  `SYNONYMES` outside a measured ablation.
- A non-neutral default: a rejected lever must stay exposed at its neutral
  value, and `Searcher()` with no arguments must still reproduce the published
  baseline (0.672 recall@10 on dev at the time of writing).
- A secret in the diff, or a write path pointing at a versioned measurement
  artefact (`docs/mesures/jalon3/reecritures.json` is a **read-only**
  reproducibility anchor; execution caches live under `data/`).

**If issues are found**, output them and **stop** — do not open or edit a PR:

```
⚠ Review found issues — PR blocked:

1. [Traceability] docs/eval-jalon4.md:82 — "6,0× the latency" no longer follows
   → F_dev.json now gives 10.164/1.831 = 5.55. Recompute or drop the factor.

2. [Protocol] scripts/ablations_jalon4.py:210 — n_probe and pool move together
   → Add the configuration isolating n_probe at neutral pool, or restate the
     decision as being about the pair.
```

## Phase 2 — Craft the PR title and description

Only when Phase 1 is clean.

**Title** — Conventional Commits, matching what CI enforces on `main`:

```
^(build|chore|ci|docs|feat|fix|perf|refactor|revert|style|test)(\([a-z0-9,-]+\))?!?: .+
```

- English, imperative, no trailing period, **72 characters maximum**.
- Scope: the area touched (`search`, `rerank`, `rewrite`, `eval`, `benchmark`,
  `corpus`, `parse`, `refs`, `integrity`, `docs`, `journal`, `plan`, `spec`).
- Describe the change as a human would. No internal task or milestone numbers,
  no `Co-Authored-By`, no Claude Code attribution of any kind.
- A measured outcome belongs in the title when there is one:
  `feat(eval): adopt LLM query rewriting — recall@10 0.672 to 0.852 on dev`.

**Description** — state what was delivered, not the process:

- The decision and the figure that justifies it, with its `p_amelioration` and
  its sample size.
- What was **rejected** and why. A measured negative result is a deliverable
  here, not a failure to hide.
- The caveats a reader needs in order not to over-read the figure.
- Never a task list, a review round count, or a plan reference.
