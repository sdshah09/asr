# Phase 1 — Measurement baseline

**Date:** 2026-08-19
**Status:** done
**Scripts:** `bench.py`, `stage_timing.py`

## Objective
Establish a trustworthy baseline: how long does transcription take, and can the number be relied on?

## Prediction
Expected RTF around 0.1-0.3 on CPU with `large-v3-turbo` int8, stable across runs.

## Setup
- M4 Pro, 24 GB, macOS. `large-v3-turbo`, int8, CPU, beam 1.
- Method: one warm-up run discarded, then 3 repeats, median reported, standard deviation reported.
- Synthetic test set (2.9s / 8.8s / 81s) plus 2 real 30s mp3 clips (44.1 kHz stereo).

```bash
.venv/bin/python bench.py
.venv/bin/python bench.py realset
.venv/bin/python bench.py realset --vad
.venv/bin/python stage_timing.py
```

## Results

### First run — unusable
```
medium   8.79s audio   10.04s proc   RTF 1.142   stdev 14.158s
```
Individual runs on the same file:
```
run 0: 8.06s    run 3: 8.99s
run 1: 16.11s   run 4: 17.31s
run 2: 13.31s   run 5: 14.15s
```

### Thread count sweep
```
threads=14 (default)  runs: 8.1 16.1 13.3 9.0 17.3 14.2   bimodal, 2x swing
threads=10 (P-cores)  runs: 3.0  3.0  3.0                 median 2.97s
threads= 4            runs: 4.9  4.9  4.9                 median 4.87s
```
Machine topology: `10 performance + 4 efficiency = 14 logical`.

### Baseline after the fix
```
synthetic:
file             audio     proc     RTF   stdev
long            81.11s   44.21s   0.545   0.815s
medium           8.79s    3.14s   0.357   0.101s
short            2.89s    2.91s   1.006   0.066s
TOTAL           92.79s   50.26s   0.542

real audio:
30 Second Explainer Videos  29.70s   3.03s   RTF 0.102
The 30-Second Video         30.12s   6.88s   RTF 0.228
TOTAL                       59.81s   9.91s   RTF 0.166   (vad on: 0.174)

peak memory 2256 MB, model load 1.19s
```

### Stage breakdown, real 30s clip
```
1. ffmpeg decode (mp3 -> numbers)   0.088s    1.8%
2. pad to 30s                       0.000s    0.0%
3. spectrogram (FFT + mel + log)    0.002s    0.0%
4. ENCODER                          1.045s   21.6%
5. DECODER (137 tokens)             3.700s   76.5%
TOTAL                               4.836s   RTF 0.161
```

## Findings

### 1. Default thread count cost 4-5x, and variance was the only clue
CTranslate2 defaults to all 14 logical cores. Work splits evenly, so every batch waits on the slowest core, and the OS migrates threads between P and E cores — producing the bimodal distribution. Pinning to 10 gave 5x speedup *and* eliminated variance. A hypothesis explaining both the level and the spread is much stronger than one explaining only the level.

**The Phase 1 lesson in one sentence:** reporting RTF 1.142 and moving on would have made every later comparison noise on top of a broken baseline.

### 2. Real audio is 3x better than synthetic (0.166 vs 0.542)
Both real clips are ~30s, filling the window almost exactly. Nearly zero padding waste.

### 3. RTF depends on word count, not just duration
Two files of the same length differed 2.2x (0.102 vs 0.228). The encoder's cost tracks duration; the decoder's tracks token count. A single aggregate RTF is a poor summary.

### 4. The decoder is 76% of runtime
27 ms per token, 137 times. mp3 decode and spectrogram together are under 2% — the parts that *look* expensive are free. This is why `turbo` shrinks the decoder (32 layers -> 4) and leaves the encoder alone.

### 5. VAD made real audio slightly slower (0.166 -> 0.174)
These clips are dense edited narration with no silence to skip, so VAD costs a little and saves nothing. On sparse audio (P0) it was up to 3.5x faster. **VAD's value is a property of the audio, not the model.**

## What I got wrong
- Predicted stable timings out of the box. The first measurements were unusable.
- Did not anticipate P/E core scheduling as a factor at all.

## Open questions
- The long file got *slower* with 10 threads (37.7s -> 44.2s) while short and medium got 3-5x faster. Optimal thread count may depend on file length. Unconfirmed — needs a proper sweep.

## Next
Phase 2 precision sweep, checking output correctness and not only latency.
