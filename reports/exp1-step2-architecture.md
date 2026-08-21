# Experiment 1, Step 2 — Where did Moonshine's parameters go?

**Date:** 2026-08-20
**Status:** done
**Scripts:** `inspect_models.py`

## Objective
Moonshine tiny is 27M, Whisper tiny.en is 38M. Which half did they shrink, and which did they protect? No timing — read the architecture from the weights.

## Prediction
"I don't know — maybe it does not handle vowels, um / I'm, something like that."

i.e. expected the smaller model to be worse at recognising filler sounds. A *quality* hypothesis rather than a parameter-location one, and testable.

Counter-evidence already on record: `moonshine-tiny` transcribed "Um" correctly in three consecutive runs (E1.0).

## Setup
```bash
.venv/bin/python inspect_models.py
```
Loads both models from HuggingFace, sums parameters per component, lists the largest tensors, and searches for positional embedding tensors in each encoder.

## Results

### Parameter split
```
                    whisper tiny.en    moonshine tiny    saved
total                   37,760,256        27,092,736    10,667,520
  encoder               10,575,360         9,671,904       903,456   ( 8%)
  decoder               27,184,896        17,420,832     9,764,064   (92%)

enc % of model               28.0%             35.7%
dec % of model               72.0%             64.3%
```

### Largest tensors
```
WHISPER tiny.en
    19,915,776  (51864, 384)     model.decoder.embed_tokens.weight
       589,824  (1536, 384)      model.encoder.layers.0.fc1.weight
       ...

MOONSHINE tiny
     9,437,184  (32768, 288)     model.decoder.embed_tokens.weight
     1,161,216  (576, 288, 7)    model.encoder.conv2.weight
       663,552  (2304, 288)      model.decoder.layers.0.mlp.fc1.weight
       ...
```

### Config comparison
```
                  whisper   moonshine
encoder layers          4           6
decoder layers          4           6
hidden size           384         288
vocab              51,864      32,768
max_source_pos       1500         n/a

positional embeddings in encoder:
  whisper   ['model.encoder.embed_positions.weight']
  moonshine NONE FOUND
```

## Findings

### 1. 92% of the savings came from the decoder; the encoder is barely touched
Encoder 10.6M -> 9.7M (8% smaller). Decoder 27.2M -> 17.4M (36% smaller). **Acoustic capability was protected; the writing half was cut.**

### 2. One tensor accounts for almost all of it
```
whisper   decoder.embed_tokens.weight   (51864, 384)  = 19,915,776
moonshine decoder.embed_tokens.weight   (32768, 288)  =  9,437,184
                                          difference  = 10,478,592
```
**10.5M of the 10.7M total savings is the vocabulary embedding table.** Everything else is rounding error.

Vocab 51,864 -> 32,768 because Whisper covers 99 languages and Moonshine tiny is English-only — it does not need tokens for Japanese, Arabic or Hindi. Hidden size 384 -> 288 shrinks the same table again.

**The design decision is "specialise, do not degrade": same ears, smaller dictionary.**

### 3. Whisper's 30-second limit is a parameter, not a setting
`max_source_positions = 1500` is a **learned tensor of shape [1500, 384]** — one slot per position, and 1500 positions is exactly 30 seconds.

You cannot feed Whisper 40 seconds because **there is no position 2001 in that table.** This is the physical cause of padding, visible in the weights.

Moonshine has no such table, so it has no length to be limited to. This is the same fact E1.1 measured as a flat ~180 ms floor.

### 4. Moonshine is deeper and narrower, with a heavier conv frontend
6 layers vs 4, hidden 288 vs 384. Its second-largest tensor is `encoder.conv2.weight (576, 288, 7)` — more work in the convolutions before attention.

## What I got wrong
The prediction aimed at the wrong half. Smaller *did* mean "worse at something", but not at hearing — the encoder is essentially unchanged. What was cut is **how many distinct tokens it can produce**.

## Open questions — directly implied by finding 2
If the cut is vocabulary rather than acoustics, Moonshine's weakness should appear on:
- rare words and proper nouns
- technical jargon and drug names (the `atorvastatin` sentence from P0)
- anything non-English

**All testable.** A vocabulary-stress test would confirm or refute the "specialise, do not degrade" reading.

## Next
**E1.3 — vocabulary stress test.** Run both models on rare words, names, jargon and the P0 medical sentence. Prediction: Moonshine matches Whisper on common English and degrades on rare tokens.
