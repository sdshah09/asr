# Reports

One report per experiment step. Each records what was run, what came back, what it means, and what turned out to be wrong.

**Why these exist separately from `ASR-notes.md`:** the notes hold understanding (how Whisper works, what the terms mean). These hold *evidence* — dated, reproducible, with the exact commands. When a later result contradicts an earlier one, both stay, and the correction is written down.

## Index

| # | Report | Status | Headline |
|---|---|---|---|
| P0 | [phase0-break-it.md](phase0-break-it.md) | done | 5 silent failure modes found; VAD fixes 3 |
| P1 | [phase1-baseline.md](phase1-baseline.md) | done | Default thread count cost 4-5x |
| P2 | [phase2-precision.md](phase2-precision.md) | done | float32 hallucinated 28s of content; int8 correct |
| E1.0 | [exp1-step0-moonshine.md](exp1-step0-moonshine.md) | done | No-padding claim confirmed by one printed shape |
| E1.1 | [exp1-step1-scaling.md](exp1-step1-scaling.md) | done | Whisper = ~180ms padding floor + ~7ms/word; Moonshine 2.6x faster at 1s, 1.27x at 30s |
| E1.2 | architecture inspection | **next** | where did Moonshine's missing 12M params go? |

## Template

```markdown
# <Phase/Step> — <one-line title>

**Date:** YYYY-MM-DD
**Status:** done / in progress / abandoned
**Scripts:** `x.py`, `y.py`

## Objective
One sentence. What question is this answering?

## Prediction
Written BEFORE running. Being wrong here is the point.

## Setup
Machine, models, audio, exact commands.

## Results
Raw numbers. Tables. No interpretation yet.

## Findings
What the numbers mean. One heading per finding.

## What I got wrong
Predictions that failed, and why. Never delete this section.

## Open questions
Things this raised that are not yet answered.

## Next
The single next step.
```

## House rules

1. **Write the prediction before running.** A report with an empty prediction section is a data dump, not an experiment.
2. **Raw numbers before interpretation.** Results and findings stay separate so someone can disagree with the reading.
3. **Never delete a wrong result.** Supersede it, keep both, explain the difference. Three findings in these reports were later corrected — that history is the most useful part.
4. **Record the command.** Future-you cannot reproduce "I ran the benchmark."
5. **If runs disagree, the disagreement is the finding.** Chase it before recording a number.
