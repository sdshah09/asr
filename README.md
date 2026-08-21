# wispr — learning ASR from zero

Notes and runnable experiments from working through how Whisper turns sound into text, and how to run it in production.

## Start here

**[ASR-notes.md](ASR-notes.md)** — the whole thing. Start with the **Revision card** at the top: the model in four steps, the numbers worth memorising, every term in one line, the measured findings, and the failure modes. One page, re-readable alone.

**[reports/](reports/)** — one report per experiment step: what was run, the raw numbers, what it means, and what turned out to be wrong. Dated and reproducible.

**[EXPERIMENTS.md](EXPERIMENTS.md)** — component breakdown and experiment matrix for building a dictation system. Nine swappable components, the alternatives for each, and what to measure.

| Part | Contents |
|---|---|
| **1** | The whole idea in plain language, 5 minutes |
| **2** | Glossary — every term, with *what it is / how it works / a real example* |
| **3** | The pipeline traced end to end with real numbers |
| **4** | Running it: CLI, Python, every decoding flag |
| **5** | Production engineering path — 7 phases, plus measured results |

Part 2 covers: sound and recording, wave-to-picture (FFT, mel, spectrograms), neural network basics, transformers and attention, prediction, Whisper specifics, and hardware/memory/inference engines.

## Scripts

| File | What it does |
|---|---|
| `trace_whisper.py` | Traces a full transcription stage by stage — real shapes, real token probabilities |
| `trace_mel.py` | Zooms in on waveform → spectrogram: FFT, mel buckets, pure-tone demo |
| `bench.py` | Benchmark — RTF, peak memory, variance across a test set |
| `make_hard_testset.py` | Builds 10 audio cases designed to make Whisper fail |
| `break_it.py` | Runs them with VAD on/off, flags hallucination / empty / looping |
| `stage_timing.py` | Where the time goes — per-stage stopwatch |
| `quant_demo.py` | Quantization on a real weight matrix, with the error measured |
| `moonshine_hello.py` | Minimal Moonshine transcription — the HuggingFace/PyTorch path |
| `scaling.py` | Duration vs time for Whisper and Moonshine, with an ASCII plot |
| `inspect_models.py` | Architecture comparison — parameter split, largest tensors, positional embeddings |

## Setup

```bash
brew install ffmpeg
uv venv .venv
uv pip install --python .venv/bin/python faster-whisper mlx-whisper
```

One environment for everything: `.venv/bin/python` runs every script here.
Note your shell default `python3` is Anaconda — `pip install` from a plain
shell lands there, not in `.venv/`, and imports will fail.

## Run

```bash
# trace the pipeline on any audio
.venv/bin/python trace_whisper.py audio.m4a

# zoom in on the spectrogram step
.venv/bin/python trace_mel.py

# benchmark (generates a synthetic test set on first run)
.venv/bin/python bench.py
.venv/bin/python bench.py realset --vad

# build the adversarial set and run it
python3 make_hard_testset.py
.venv/bin/python break_it.py
```

## Measured baseline

M4 Pro, 24 GB, CPU only, `large-v3-turbo`, synthetic test set:

| test set | config | RTF | speed |
|---|---|---|---|
| real audio (2x 30s mp3) | int8, vad off | **0.166** | 6.0x |
| real audio | int8, vad on | 0.174 | 5.8x |
| synthetic (2.9-81s) | int8 | 0.542 | 1.8x |
| synthetic, medium clip | float32 | 0.266 | — |

Three findings that contradict common advice, all measured here:

1. **Default thread count cost 4-5x.** CTranslate2 uses all 14 logical cores; pinning to the 10 performance cores dropped an 8s clip from 8-17s to a steady 3.0s.
2. **float32 beats int8 on Apple Silicon** (24% faster, identical transcripts). The "int8 is faster" rule assumes memory bandwidth is the bottleneck. It isn't here.
3. **VAD made real audio slightly slower** (0.166 -> 0.174). These clips are dense edited speech with no silence to skip, so VAD costs a little and saves nothing. On sparse audio it was up to 3.5x faster and fixed both hallucination and duplication.

`faster-whisper` on Mac is CPU-only — the GPU sits idle. `mlx-whisper` uses it.

## Next

- [ ] Replace `testset/` with real recordings — noisy, accented, multi-speaker
- [ ] "Break it deliberately" — log every failure mode; that list becomes the eval harness
- [ ] Phase 2: full sweep (model × precision × VAD)
- [ ] Phase 6: eval harness with sliced WER
