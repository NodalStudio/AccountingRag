<!--
PR title must be a Conventional Commit — it becomes the commit message on main.
CI checks it: type(scope): subject, 72 characters max, English, imperative.
Run /conventional to review the diff against this repo's measurement invariants
and produce the title.
-->

## What this delivers

<!-- The decision and the figure that justifies it. One or two sentences. -->

## Measurement

<!--
Delete this section only if the change touches no measured behaviour.

| config | recall@5 | recall@10 | MRR | split |
|---|---|---|---|---|
|  |  |  |  |  |

- Paired bootstrap vs reference: delta, IC95, p_amelioration, worst per-category loss
- Adopted / rejected, against the criterion (p >= 0.95 on recall@10, no category below -0.05)
- Versioned artefact: docs/mesures/...
-->

## What was rejected

<!--
A measured negative result is a deliverable here. State it, with its figure.
Delete only if nothing was measured and rejected.
-->

## Caveats

<!--
What a reader needs in order not to over-read the figures: sample size,
attainability versus recall, unmeasured cost, device for any latency.
-->

## Checklist

- [ ] Every published figure recomputes from a JSON under `docs/mesures/`, derived figures included
- [ ] One variable at a time; any configuration moving two parameters has its isolating counterpart
- [ ] The adoption criterion was not substituted after measuring
- [ ] The frozen split chose no parameter
- [ ] Any new control has a test that makes it fail (mutation applied and reverted)
- [ ] No mechanism asserted without the measurement behind it
- [ ] `Searcher()` with no arguments still reproduces the published baseline
- [ ] The same claim is consistent across `README.md`, `docs/eval-*.md` and `JOURNAL.md`
- [ ] No secret, and no write path pointing at `docs/mesures/`
