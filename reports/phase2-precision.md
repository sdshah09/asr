# Phase 2 — Precision sweep (int8 vs float32)

**Date:** 2026-08-19
**Status:** done — supersedes an earlier conclusion from the same day
**Scripts:** ad-hoc (see commands), `quant_demo.py`

## Objective
Which `compute_type` should production use? Measure latency, memory, and output correctness.

## Prediction
int8 faster than float32 (the standard claim: quantization reduces memory traffic, which is the bottleneck), with a small accuracy cost.

## Setup
M4 Pro, CPU, `large-v3-turbo`, 10 threads, beam 1. Two test files: an 8.8s synthetic clip and a real 30.12s mp3.

## Results

### Supported compute types on ARM CPU
```
int8           ok
int8_float32   ok
float32        ok
int8_float16   unsupported
float16        unsupported
bfloat16       unsupported
int16          unsupported
```

### First measurement — synthetic clip only
```
precision      median     RTF   peakMB   transcript
int8            3.07s   0.349     2041   identical
float32         2.34s   0.266     4456   identical
```
Conclusion drawn at the time: **float32 is 24% faster on Apple Silicon.** This turned out not to generalise.

### Second measurement — both file types
```
file            ct        words   median     RTF
synthetic 8.8s  int8         23    2.86s   0.325
synthetic 8.8s  float32      23    2.12s   0.242
real 30s        int8        105    6.64s   0.221
real 30s        float32     216   24.36s   0.809
```

**The word counts differ.** Same audio, 105 vs 216 words.

### Transcripts compared
```
int8      last segment: [29.00 -> 30.00] "Oh."                  <- ends with the audio
float32   last segment: [55.00 -> 58.00] "I'll do it, but..."   <- 28s past the end
```
float32's invented segments:
```
[30.00 -> 32.00] Let me show you.
[32.00 -> 34.00] I'm Wrestle to win it right now.
[38.00 -> 39.00] Why, you won't be at that place.
[50.00 -> 53.00] It's an angle of art.
[55.00 -> 58.00] I'll do it, but you'll have time to将 it out.
```

### VAD as a fix — rejected
```
int8      vad=off    6.50s  105 words  ends 30.00s  clean
int8      vad=on     6.64s  105 words  ends 30.00s  clean
float32   vad=off   24.89s  165 words  ends 58.00s  HALLUCINATED
float32   vad=on    27.15s  216 words  ends 58.00s  HALLUCINATED
```

### Temperature fallback isolated
```
float32, temperature=[0.0] (greedy, no fallback):
  5.67s / 5.82s / 5.95s   — 188 words every run, deterministic

float32, default temperature ladder:
  17.52s / 23.71s / 25.16s — 116 / 196 / 182 words, different every run
```

## Findings

### 1. float32 hallucinated 28 seconds of content that does not exist
The file is 30.12s. The window is 30s, so the second chunk holds 0.12s of audio and 29.88s of padding — near-pure silence. int8 emitted nothing for it; float32 hallucinated into it. Same failure mode as P0 case 01. One invented segment contains a Chinese character.

### 2. VAD does not fix it
Hypothesis tested and rejected. float32 with `vad_filter=True` still ran to 58.00s and produced *more* words.

### 3. The anti-looping safeguard causes instability and costs 3-4x
This audio is a novelty song repeating "30 seconds long". The compression-ratio check reads genuine repetition as a stuck model and retries at higher temperature — i.e. random sampling. Different output every run. **The safeguard against broken output was itself producing broken output.** For reproducible transcripts: `temperature=[0.0]`.

### 4. Per token, float32 really is faster here
```
float32 greedy:  188 words in 5.67s  ->  30 ms/word
int8:            105 words in 6.50s  ->  62 ms/word
```
But 83 of those words were invented. **Faster at producing the wrong answer.**

### 5. Precision changes the decoding path, not just speed
Tiny numerical differences flip threshold decisions (is this silence? is this repetitive?) which cascade into completely different output. Not documented in any comparison table — only appears on your own audio.

### 6. Quantization internals (from `quant_demo.py`)
Real 5120x1280 matrix, 6.5M weights:
```
float32 26.2 MB -> int8 6.6 MB + 0.02 MB of scales  =  3.99x smaller
weights off by 1.95%, layer OUTPUT off by only 1.14%
```
Errors partially cancel across a dot product's thousands of terms. Weights below one quantization step round to zero and vanish entirely; large weights survive nearly exactly.

## What I got wrong
- **Concluded "float32 beats int8 on Apple Silicon" from one 8-second synthetic file.** It did not survive real audio. The synthetic clip had no trailing partial chunk, so no hallucination trigger existed.
- **Predicted VAD would fix the hallucination.** It did not.
- **Described int8's overhead as "unpack every weight."** Wrong — int8 GEMM accumulates in int32 and rescales once per output, never unpacking weights to float.

## Decision
**Use int8.** Not because it is faster per token — it is not — but because on real audio it stayed correct and finished in 6.5s where float32 took 25s producing fabricated content.

## Open questions
- Does float32 hallucinate on files that are exact multiples of 30s (no partial trailing chunk)?
- Would `int8_float32` behave like int8 or like float32 on the threshold decisions?

## Next
Add an "output extends past audio duration" check to `break_it.py` — it would have caught this automatically.
