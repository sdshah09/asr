# wispr — learning ASR from zero

Notes and runnable experiments from working through how Whisper turns sound into text, and how to run it in production.

## Start here

**[ASR-notes.md](ASR-notes.md)** — the whole thing, ~1000 lines.

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
| `bench.py` | Phase 1 benchmark — RTF, peak memory, variance across a test set |

## Setup

```bash
brew install ffmpeg
uv tool install mlx-whisper                        # Apple GPU path, for the traces
uv venv .venv && uv pip install --python .venv/bin/python faster-whisper
```

## Run

```bash
# trace the pipeline on any audio
~/.local/share/uv/tools/mlx-whisper/bin/python trace_whisper.py audio.m4a

# zoom in on the spectrogram step
~/.local/share/uv/tools/mlx-whisper/bin/python trace_mel.py

# benchmark (generates a synthetic test set on first run)
.venv/bin/python bench.py
.venv/bin/python bench.py --compute-type float32 --vad
```

## Measured baseline

M4 Pro, 24 GB, CPU only, `large-v3-turbo`, synthetic test set:

| config | RTF | peak mem |
|---|---|---|
| int8, 10 threads | 0.542 | 2256 MB |
| float32, 10 threads | 0.266* | 4456 MB |

\* on the medium clip. **float32 is faster than int8 on Apple Silicon** — the opposite of the usual rule, because bandwidth is not the bottleneck here. See Part 5 for why.

Two caveats on every number above: the test set is synthetic TTS (unrealistically clean), and `faster-whisper` on Mac is CPU-only. Replace with real audio before drawing conclusions.

## Next

- [ ] Replace `testset/` with real recordings — noisy, accented, multi-speaker
- [ ] "Break it deliberately" — log every failure mode; that list becomes the eval harness
- [ ] Phase 2: full sweep (model × precision × VAD)
- [ ] Phase 6: eval harness with sliced WER
