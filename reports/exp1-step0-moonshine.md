# Experiment 1, Step 0 — Get Moonshine running

**Date:** 2026-08-20
**Status:** done
**Scripts:** `moonshine_hello.py`

## Objective
Transcribe one file with Moonshine. No timing, no comparison. Prove the install works and learn the API shape.

## Why Moonshine
Whisper pads every input to 30 seconds, which is why `large-v3-turbo` took 2919 ms on a 4.3s clip (P1/E-latency). Moonshine claims compute that scales with actual audio length — a direct fix for the measured problem.

Models available (all MIT, moonshine-ai):
```
moonshine-tiny              27M     (Whisper tiny is 39M)
moonshine-base              61M     (Whisper base is 74M)
moonshine-streaming-tiny    44M
moonshine-streaming-small  140M
moonshine-streaming-medium 266M
```

## Setup
```bash
uv pip install --python .venv/bin/python "transformers>=4.48"    # transformers 5.15.1, torch 2.13.0
.venv/bin/python moonshine_hello.py
.venv/bin/python moonshine_hello.py dictation.aiff moonshine-ai/moonshine-base
```
Audio: `dictation.aiff`, 4.30s — *"um so I think we should probably ship this on Friday and then tell the team"*.

## Results
```
audio: dictation.aiff  (4.30s, 68,864 samples)
model input shape: (1, 68864)   <- NOT padded to 30s
transcript: Um so I think we should probably ship this on Friday and then tell the team
```

Determinism check, 3 consecutive runs — byte-identical output each time. `generation_config.json` has no sampling enabled.

## Findings

### 1. The no-padding claim is confirmed by a printed shape
```
Moonshine input : (1, 68864)    <- the actual sample count
Whisper input   : (1, 480000)   <- always, regardless of audio length
```
No benchmark required. The architectural difference is visible before measuring anything.

### 2. Third inference runtime in this repo, with different conventions
| runtime | used by | API |
|---|---|---|
| CTranslate2 | faster-whisper | `model.transcribe(path)` handles everything |
| MLX | mlx-whisper | `mlx_whisper.transcribe(path)` |
| PyTorch/transformers | Moonshine | processor -> tensors -> `generate()` -> decode |

The third is the standard HuggingFace pattern and transfers to most models on the Hub.

### 3. `generate()` is the decoder loop
Audio -> tensors -> `generate()` -> **token IDs** -> decode -> text. Same loop traced in `trace_whisper.py`, wrapped in one call. `max_new_tokens` is the runaway guard — exactly the behaviour float32 exhibited in P2.

### 4. Type mismatch, solved by printing the type
`mlx_whisper.audio.load_audio` returns an **mlx array**; transformers wants numpy or torch. Fixed with `np.array(...)`. Most confusion in this area is assuming something is numbers when it is an object.

### 5. Loading weights every run is the worker-pool problem, felt directly
Each `python script.py` is a fresh process with empty memory, so weights are re-read from disk. Download happens once (cached in `~/.cache/huggingface`); the load happens every run. **`scaling.py` must load each model once and loop inside**, or it will measure load time instead of inference time.

## What I got wrong
Reported that `moonshine-tiny` mangled "Um" as "I'm", from a **single run**. Three repeats show it gets it right. Same error as the P2 float32 conclusion: one measurement, confident claim, wrong. Twice in one day.

One unexplained detail: that first run produced genuinely different text (`"I'm so I think ... team."` with quotes and a full stop) on the same model and input, while generation is demonstrably deterministic now. Most likely a partially-downloaded or different snapshot. **Flagged rather than explained away.** If it recurs, chase it.

## Open questions
- Does `moonshine-base` (61M) beat `whisper base.en` (74M) on quality, not just size?
- Do the streaming variants need `transformers` from git, and does that break the current environment?

## Next
**E1.1 — the scaling test.** Clips at 1/2/5/10/20/30s through both Whisper and Moonshine, plotting duration vs time.

Prediction to be written before running:
- Whisper `tiny.en`: expected roughly flat (padding dominates)
- Moonshine tiny: expected roughly linear through the origin
- Crossover point unknown
