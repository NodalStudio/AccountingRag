# AccountingRAG — working instructions

Open-source French accounting RAG, built on public sources only. North star:
sitting real DSCG UE4 ("Comptabilité et audit") papers. Deliverable order is
deliberate — **structured corpus and benchmark first, RAG as the demo on top**.

Full design: `docs/superpowers/specs/2026-08-14-accountingrag-design.md`.
Engineering journal (decisions, mistakes, blog material): `JOURNAL.md`.

## The measurement laws

This repo's entire value is the credibility of its numbers. A wrong figure here
is worth less than no figure. Three review passes on milestone 3 found zero
wrong result figures and a dozen wrong statements *about* them — so these laws
bind prose as tightly as code.

1. **Every published figure is recomputable** from a JSON under `docs/mesures/`,
   produced by a versioned script. That includes *derived* figures — a factor, a
   duration, a count. When a measurement is re-run, grep the other documents for
   the old number.
2. **One variable at a time.** An ablation that moves two parameters measures
   their sum. If a second must move, the configuration isolating the first
   belongs in the same grid.
3. **Adoption criterion, fixed before measuring:** `p_amelioration >= 0.95` on
   **recall@10** (paired bootstrap, `n_boot=10000`, `seed=42`) *and* no category
   losing more than 0.05. Never substitute a metric that flatters the result
   after the fact. A measured negative result is a deliverable.
4. **`benchmark/test.jsonl` is frozen.** Never used to choose a parameter; one
   execution per milestone closing, and the figure is published as measured.
5. **A control nobody has seen fail proves nothing.** Every integrity,
   freshness or reproduction control needs a test that makes it fail. This repo
   once shipped a reproduction control that validated the bug it was meant to
   catch, because it demanded agreement with a figure nobody re-derived.
6. **Never publish a mechanism before running the control that settles it.**
   Three plausible mechanisms shipped in milestone 3; all three were wrong while
   their conclusions were right. The control usually costs three minutes.
7. **A latency is not a property of the system** — it is a property of (machine,
   device, load). Quote the device, and quote orders of magnitude rather than
   decimals: four runs of the same probe spanned ×66 to ×106.
8. **Neutral defaults.** A rejected lever stays exposed at its neutral value, so
   `Searcher()` with no arguments reproduces the published baseline.
9. **Benchmark integrity.** A rewriter, generator or judge sees only the
   question text — never the gold citations, the corpus, or the retrieval
   results. A question derived from the document that answers it must be held
   out of the measurement index, or the measurement is circular.
10. **`docs/mesures/**` is read-only at runtime.** Those files are
    reproducibility anchors. Execution caches live under `data/` and are
    gitignored (`Rewriter(ecrire_cache=False)` enforces this).

## Git workflow

Two commit formats, by branch:

- **Feature branches: pure gitmoji** — `✨ Add abstention family to the benchmark`
  (one emoji, capitalized one-line subject, English, no conventional prefix).
  See `.claude/commands/gitmoji.md`.
- **`main`: Conventional Commits only.** CI enforces
  `^(build|chore|ci|docs|feat|fix|perf|refactor|revert|style|test)(\([a-z0-9,-]+\))?!?: .+`
  on every commit pushed to `main` and on every PR title. Subject ≤ 72
  characters. See `.claude/commands/conventional.md`.
- **Never any attribution.** No `Co-Authored-By`, no "Generated with Claude
  Code", no mention of the tool in commits, PR titles, descriptions or issues.
  `.claude/settings.json` sets `includeCoAuthoredBy: false`; do not reintroduce
  it by hand.
- Commit messages describe the change as a human would: no internal task
  numbers, no plan references, no review-round counts.

## Language convention

- **Developer-facing → English**: identifiers, comments, commit messages, PR
  titles and descriptions, issues.
- **Documentation and reports → French** (`README.md`, `docs/eval-*.md`,
  `JOURNAL.md`, benchmark questions). This is a French accounting project; its
  readers are French practitioners.
- **Regulatory vocabulary stays in French, always** — `amortissement`,
  `provision réglementée`, `mali de fusion`. Never translate an accounting term
  in code, data or prose: the citation must match the source text.

## Commands that matter

```sh
uv run python scripts/download_data.py     # fetch the Recueil PDF (sha256 checked)
uv run python scripts/build_corpus.py      # PDF -> data/corpus.db + anomaly report
uv run python scripts/build_index.py       # chunks + FTS5 + sqlite-vec (~14 min)
uv run pytest -q                           # full suite
uv run python scripts/run_eval.py --mode all --split dev
uv run python scripts/audit_reecritures.py # benchmark integrity, seconds, free
```

`scripts/cloture_jalon3.py` **consumes the frozen split and calls a paid API** —
run it only at a milestone closing.

## Never

- Rebuild the index inside a milestone that declared no rebuild.
- Touch `SYNONYMES` outside a measured ablation (a wrong synonym is worse than a
  missing one — an early entry conflated `amortissement dégressif` with
  `amortissement dérogatoire`, which are not the same thing).
- Mix `nature: comptable` and `nature: fiscale` in an index without the tag: a
  RAG that does answers "not deductible" to someone who asked "depreciable".
- Commit a secret. `.env` is gitignored; `.env.example` carries a placeholder.
