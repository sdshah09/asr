# Experiment 1, Step 1 — Does cost scale with audio length?

**Date:** 2026-08-20
**Status:** done
**Scripts:** `scaling.py`

## Objective
Whisper pads every input to 30 seconds; Moonshine claims it does not. Measure transcription time against audio length for both and see the shape.

Motivation: `large-v3-turbo` took 2919 ms on a 4.3s clip (RTF 0.68) — nearly slower than real time. Padding is the suspected cause.

## Prediction
Written before running.

- **Whisper `tiny.en`:** same time for 1s and 30s. Padding means it always does 30s of work.
- **Moonshine tiny:** different across lengths, because it does not pad.
- Crossover: unknown.

## Setup
M4 Pro, CPU, 10 threads. Clips cut from one continuous `say` recording to 1/2/5/10/20/30s, 16 kHz mono.

- Whisper `tiny.en` via faster-whisper / CTranslate2, **int8**
- Moonshine tiny via transformers / PyTorch, **float32**

Each model loaded **once**, all clips looped inside the same process. Warm-up run discarded, 3 repeats, median.

```bash
.venv/bin/python scaling.py
```

## Results
```
 audio  words    whisper   moonshine    w RTF    m RTF
    1s      2       183ms         71ms    0.183    0.071
    2s      4       205ms         65ms    0.103    0.033
    5s      9       249ms        100ms    0.050    0.020
   10s     22       299ms        152ms    0.030    0.015
   20s     48       539ms        353ms    0.027    0.018
   30s     69       663ms        520ms    0.022    0.017

1s -> 30s:  whisper   183ms -> 663ms  = 3.61x   (audio grew 30x)
            moonshine  71ms -> 520ms  = 7.35x
```

## Findings

### 1. Whisper is NOT flat — the prediction was half wrong
183 ms -> 663 ms, a 3.6x increase. **Only the encoder is padded. The decoder scales with word count.**
```
 1s clip ->  2 words
30s clip -> 69 words

extra time  = 663 - 183 = 480 ms
extra words =  69 -   2 =  67
per word    = 480 / 67  ~ 7.2 ms

predicted 30s = 180ms floor + (69 x 7.2ms) = 677 ms
measured                                   = 663 ms   OK
```
**Whisper's cost = a flat ~180 ms floor + ~7 ms per word.** The floor is the padding tax and never goes away, even for a 1-second clip. This is the encoder/decoder split from `stage_timing.py` appearing as the shape of a curve.

### 2. Moonshine scales more, as predicted
7.35x versus Whisper's 3.61x — it does real work proportional to the audio rather than padding it out.

### 3. The advantage shrinks with length — predicted by neither of us
```
 1s:  183 vs  71 ms  ->  Moonshine 2.6x faster
30s:  663 vs 520 ms  ->  Moonshine 1.27x faster
```
The lines converge. Moonshine's win is **entirely** the removal of padding waste, and that waste shrinks toward zero as audio approaches 30 seconds. At 30s both models do the same real work.

**Practical consequence:** Moonshine is a large win for dictation (2-10s utterances) and nearly irrelevant for long-form transcription. The RTF 1.25-at-2.3s finding from Phase 0 is exactly the regime it fixes.

### 4. The comparison is unfair to Moonshine
Whisper ran on CTranslate2 (optimised C++, int8). Moonshine ran on PyTorch (float32, unoptimised). **Moonshine wins while carrying the slower runtime.** An ONNX or GGUF build would likely widen the gap.

## What I got wrong
Predicted Whisper would be flat. It is flat *in the encoder* only — the decoder's per-word cost dominates at longer durations. The padding intuition was right but applied to the whole model instead of half of it.

## Open questions
- What does the curve look like past 30s, where Whisper starts chunking? Expect a step at each 30s boundary.
- Does `moonshine-base` (61M) keep the short-clip advantage, or does the extra size eat it?
- How much faster is Moonshine on an ONNX/GGUF runtime, i.e. what is the fair comparison?

## Next
**E1.2 — architecture inspection.** No timing. Print both model structures, look for positional embeddings in each encoder, read the attention window from the configs, and split parameter counts by component. Question: Moonshine tiny is 27M and Whisper tiny is 39M — where did the 12M go, and what did they protect?
