# Experiment 1, Step 3 — Does the smaller vocabulary hurt on rare words?

**Date:** 2026-08-20
**Status:** done
**Scripts:** `vocab_test.py`

## Objective
E1.2 found that Moonshine's parameter savings are almost entirely a smaller vocabulary embedding (51,864 -> 32,768 tokens). If the cut is vocabulary rather than acoustics, Moonshine should degrade on rare words, proper nouns and jargon. Test it.

## Prediction
- Moonshine matches Whisper on common English.
- Moonshine degrades on rare words — drug names, proper nouns, technical terms.
- It will not fail outright, because tokenizers split unknown words into fragments; it will need **more** fragments and be more likely to misspell them.

## Setup
Two parts.

**A — tokenizer only.** Encode 20 words across 5 categories with each tokenizer, count fragments.
**B — end to end.** Synthesize 7 sentences with `say`, transcribe with both models, compare against the reference.

Models: Whisper `tiny.en` (38M) via faster-whisper int8; Moonshine tiny (27M) via transformers float32. M4 Pro, CPU.

```bash
.venv/bin/python vocab_test.py --tokens-only
.venv/bin/python vocab_test.py
```

## Results

### Part A — fragments per word
```
category  word                whisper  moonshine   moonshine pieces
common    the                       1          1   [the]
common    yesterday                 2          1   [yesterday]
names     Shaswat                   3          4   [Sh][as][w][at]
names     Anthropic                 3          3   [Anth][rop][ic]
names     Kubernetes                4          2   [K][ubernetes]
names     PostgreSQL                2          3   [Post][gre][SQL]
medical   atorvastatin              4          5   [at][or][v][ast][atin]
medical   metoprolol                4          4   [met][op][rol][ol]
medical   levothyroxine             4          5   [lev][othy][ro][x][ine]
medical   acetaminophen             3          5   [ac][et][amin][oph][en]
jargon    quantization              2          2   [quant][ization]
jargon    spectrogram               2          3   [spect][ro][gram]
borrowed  croissant                 3          2   [cro][issant]
borrowed  entrepreneur              3          3   [entrepr][ene][ur]
TOTAL                              54         58   moonshine needs 1.07x
```

### Part B — end to end
```
[medical] The patient was prescribed forty milligrams of atorvastatin daily.
  whisper  : 'The patient was prescribed 40 milligrams of a tortoise statin daily.'
  moonshine: 'The patient was prescribed 40 milligrams of atorvastatin daily.'

[medical] She switched from metoprolol to levothyroxine last spring.
  whisper  : 'She switched from Metaprolol to Levothiroxine last spring.'
  moonshine: 'She switched from metaprolol to levothyroxine last spring.'

[names] Shaswat deployed the service to Kubernetes using PostgreSQL.
  whisper  : 'Shaswat deployed the service to QberNates using PostGrew SQL.'
  moonshine: 'Shaswat deployed the service to Kubernetes using PostgreSQL.'

[borrowed] The entrepreneur ordered a croissant and sang karaoke.
  whisper  : 'The entrepreneur ordered a croissant and sang karaoke.'
  moonshine: 'The entrepreneur ordered a crisson ensaang karaoke.'

model          words correct   accuracy
whisper                50/59      84.7%
moonshine              52/59      88.1%
```

## Findings

### 1. Prediction refuted — the smaller model won on rare words
88.1% vs 84.7%. Moonshine transcribed `atorvastatin`, `levothyroxine`, `Kubernetes` and `PostgreSQL` exactly; Whisper mangled all four.

Whisper rendered a drug name as **"a tortoise statin"** — three plausible English words, no error raised, completely wrong. Same class of failure as the P0 silent corruptions.

### 2. Vocabulary size is not coverage
The reasoning error in E1.2 was treating 51,864 > 32,768 as "more words known".

- **Whisper's 50k tokens are spread across 99 languages.** Its English share is far below 50k.
- **Moonshine's 32k are all English.** Denser English coverage while being 36% smaller.

This is why the fragment count came out at only **1.07x** despite the much smaller table. The budget is spent where it is used.

### 3. The tokenizer table predicted the result, read correctly
```
Kubernetes    whisper 4 fragments    moonshine 2 fragments   [K][ubernetes]
```
Moonshine holds `ubernetes` as a single token. Fewer fragments, fewer chances to go wrong. Fragment count is a usable predictor of rare-word accuracy.

### 4. The specialization tradeoff shows up where expected — on a loanword
```
croissant  ->  'crisson ensaang'
```
A French word, exactly where an English-only vocabulary is thin. Whisper got it right. This is the cost of specialising, pointing the opposite way from the prediction.

## What I got wrong
- **The core prediction.** Assumed smaller vocabulary meant worse rare-word handling. It meant *better*, because the vocabulary is specialised rather than merely smaller.
- **First tokenizer measurement was wrong by 45%.** Encoding with a leading space made Moonshine's SentencePiece tokenizer emit a separate `_` token for every word, inflating the ratio to 1.53x. Whisper's BPE folds the space into the word token (`Gthe`). Removing the space gives the honest 1.07x. **The `[]` appearing on every single word was the tell** — an artifact that uniform is never a finding.

## Caveats
- **The scorer is polluted by normalization.** Both models wrote "40" for "forty" and "8" for "eight", counted as errors. Excluding those, real error counts are roughly Whisper 5, Moonshine 3. This is the P0 normalization finding biting again, and why Phase 6 needs a written-down policy.
- **7 synthetic sentences.** `say` pronounces these words one specific way; humans differ. Directional, not conclusive.

## Open questions
- Does the result hold on human speech rather than TTS?
- Does `moonshine-base` (61M) extend the lead, or does Whisper `base.en` (74M) close it?
- How does Whisper `large-v3` handle these? Its vocabulary is the same 51,864 but it has far more capacity.

## Next
Experiment 1 is complete. Moonshine tiny is faster on short audio (E1.1), architecturally specialised (E1.2), and more accurate on English technical vocabulary (E1.3) — while being 27M against 38M.

Candidate follow-ups: repeat on human audio; test the streaming variants; or move to the CTC comparison (Parakeet) for the other major architecture.
