# Phase 0 — Break it deliberately

**Date:** 2026-08-19
**Status:** done
**Scripts:** `make_hard_testset.py`, `break_it.py`

## Objective
Find where Whisper fails, before optimising anything. The failure list becomes the eval test set later.

## Prediction
Expected failures on: silence (hallucination), overlapping speakers, stereo downmix, very quiet speech, heavy noise, fast speech, clipping.

## Setup
- Machine: M4 Pro, 24 GB, macOS
- Model: `large-v3-turbo`, int8, CPU, 10 threads, beam 1
- 10 synthetic adversarial cases, each run twice (VAD off / on)
- Test sentence chosen to be medically consequential: *"The patient was prescribed forty milligrams of atorvastatin daily."*

```bash
python3 make_hard_testset.py
.venv/bin/python break_it.py
```

## Results

Full log: `hardset/RESULTS.md`. Cases: `hardset/CASES.md`.

```
01_silence_30s        [vad off]  'Thank you.'                    2.8s
01_silence_30s        [vad on ]  ''                              0.8s
02_speech_in_silence  [vad off]  sentence written TWICE          6.0s
02_speech_in_silence  [vad on ]  sentence written once           3.1s
03_very_quiet         [both]     correct                         ~2.9s
04_noisy              [both]     correct                         ~3.0s
05_overlapping        [both]     speaker B absent entirely       ~2.9s
06_stereo_two_speakers[both]     'atorvastatin data' (not daily) ~2.8s
07_fast_speech        [both]     correct                         ~2.9s
08_music_no_speech    [vad off]  '.'                             3.1s
08_music_no_speech    [vad on ]  ''                              0.0s
09_long_repeated      [vad off]  correct                        33.2s
09_long_repeated      [vad on ]  correct                        28.2s
10_clipped            [both]     correct                         ~2.9s
```

RTF by audio length:
```
07_fast_speech      2.3s audio  ->  RTF 1.25
03/04/05/10         3.9s audio  ->  RTF ~0.75
09_long_repeated  160.9s audio  ->  RTF 0.21
```

## Findings

### 1. Silence produces invented text
30 seconds of digital silence returned `'Thank you.'` Trained on 680k hours of web video where clips end with "thank you"; given no acoustic evidence it falls back on that prior. VAD returns empty and is 3.5x faster.

### 2. A sentence present once was transcribed twice
The audio contains the sentence once, at t=20s. With VAD off it was emitted twice. The sentence sits near a 30s chunk boundary, appears in two chunks, and `condition_on_previous_text` carries it forward.

**Most dangerous failure found.** A duplicated dosage line in a medical transcript is serious, and nothing about the output looks wrong. VAD fixes it.

### 3. Stereo downmix replaced a word
`'daily'` became `'data'`. Not dropped — replaced with a different real word. Silent corruption is worse than an obvious failure. Fix: split channels before transcribing.

### 4. An entire speaker vanished
Speaker B's sentence is absent with no warning. Whisper has no concept of multiple speakers. Needs a separate diarization model.

### 5. Normalization is inconsistent
The same spoken sentence transcribed as `'40 milligrams'` in some cases and `'40 mg'` in others, decided by acoustic conditions. Will break WER measurement unless normalized.

### 6. RTF is monotonic with file length
6x spread from input length alone, because every file pays the full 30-second padding cost. **Batching is existential for short-clip workloads, not an optimisation.**

## What I got wrong

- **Predicted very quiet speech (4% volume) would be dropped by VAD.** It transcribed perfectly. Silero is more sensitive than assumed.
- **Predicted noise, clipping and 320 wpm would degrade output.** All three were perfect.
- **Five of ten "adversarial" cases were not adversarial.** Whisper's robustness to audio *quality* is better than expected. The real failures cluster around silence, chunk boundaries, and multiple speakers — not degradation.
- **My own `LOOPING` detector missed finding 2** (threshold was 3 repetitions; this repeated twice). Known false negative in the harness.

## Open questions
- Does a genuinely quiet human speaker (not a scaled synthetic) get dropped by VAD?
- Does the duplication reproduce at other chunk-boundary offsets?

## Next
Fix the loop detector threshold and add an "output extends past audio duration" check — that would have automatically caught the float32 failure in P2.
