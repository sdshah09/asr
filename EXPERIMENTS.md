# Dictation system — component breakdown and experiment matrix

The system is seven components. Each one is independently swappable and independently measurable. This document lists the alternatives for each, what to measure, and what is expected to matter.

**The design rule:** define the interface between components first, so any one can be swapped without touching the others. Then every experiment is "change one component, re-run the harness, compare."

---

## The pipeline

```
[1 capture] -> [2 endpoint] -> [3 stream strategy] -> [4 ASR model] -> [5 format] -> [6 personalize] -> [7 inject]
                                        |                  |
                                   [runtime]          [precision]
```

## What to measure, everywhere

| Metric | Definition | Why |
|---|---|---|
| **t_release→text** | hotkey release to text appearing | the number the user feels. THE metric. |
| **t_stage** | latency of one component alone | tells you where to optimize |
| **p50 / p95** | median and tail | p95 is what makes it feel unreliable |
| **WER** | word error rate vs reference | ASR quality |
| **MER** | *meaning* error rate — did the final text say what you meant | the metric that actually matters after an LLM pass |
| **over-edit rate** | how often the LLM changed meaning | the LLM pass's failure mode |
| **peak RSS** | memory | decides what can run resident |
| **cold start** | first-use latency | model load; kills the experience if unwarmed |

**Baseline to beat:** measured on this machine, 4.3s utterance — `tiny.en` CPU 174 ms, `turbo` GPU 452 ms, `turbo` CPU 2919 ms.

---

## Component 1 — Audio capture

| Axis | Options | Notes |
|---|---|---|
| Library | `sounddevice`, `PyAudio`, `ffmpeg -f avfoundation` | sounddevice is simplest, callback-based |
| Chunk size | 20 / 50 / 100 / 200 ms | smaller = lower latency, more overhead |
| Buffer | fixed array vs ring buffer | ring buffer needed for streaming |
| Format | always 16 kHz mono float32 | match Whisper's input; avoid a conversion step |

**Measure:** callback jitter, dropped frames, capture-to-buffer latency.
**Expected impact: LOW.** Get it working, move on. Capture is not the bottleneck.

---

## Component 2 — Endpointing (when did they stop?)

| Approach | How | Trade-off |
|---|---|---|
| **Push-to-talk** | user holds a key | zero error, zero latency, requires a key held. **Start here.** |
| Energy threshold | volume drops below X for Y ms | trivial, fails on quiet speakers and noisy rooms |
| **Silero VAD** | neural speech detector | already in faster-whisper, ~1 MB, robust |
| WebRTC VAD | classic DSP-based | very fast, less accurate |
| Semantic endpointing | LLM decides "is this a complete thought" | best UX, adds latency |

**Measure:** false cut rate (truncated mid-sentence), hang time (ms of silence before finalizing).
**Expected impact: MEDIUM.** Push-to-talk sidesteps it entirely for v1. Matters a lot for hands-free.

---

## Component 3 — Streaming strategy (the biggest lever)

This is architecture, not model choice, and it dominates perceived latency.

| Strategy | How | Latency after release | Difficulty |
|---|---|---|---|
| **Batch on release** | record, then transcribe the whole thing | full ASR cost (174–450 ms) | trivial |
| **Sliding window** | re-transcribe an overlapping window every N ms | ~window tail only | medium; stitching causes duplicates |
| **VAD-segmented** | transcribe each speech segment as it completes | ~last segment | medium |
| **True streaming (CTC)** | frame-synchronous model, emits as it goes | ~80 ms | needs a different model |
| **Speculative decoding** | small model drafts, big model verifies | 2–3x faster decode, identical output | medium |

**The point:** with streaming, most of the utterance is already transcribed when the key is released. Only the tail remains.
```
batch     : 4.3s speech -> ~450 ms after release
streaming : 4.3s speech ->  ~80 ms after release
```
Same model, same machine. **Architecture beats model size here.**

**Watch out:** stitching overlapping windows is exactly what produced the duplicated-sentence bug in the break tests. Any sliding-window implementation needs a dedup step and a test for it.

**Measure:** t_release→text, duplication rate, words lost at seams.
**Expected impact: HIGHEST.**

---

## Component 4 — ASR model

Measured on this machine, 4.3s clip, int8, CPU unless noted:

| model | latency | notes |
|---|---|---|
| `tiny.en` | **174 ms** | dropped the "um" — smaller model, less faithful |
| `base.en` | **345 ms** | good balance |
| `distil-small.en` | 661 ms | |
| `small.en` | 967 ms | |
| `large-v3-turbo` (GPU/mlx) | **452 ms** | best quality inside budget |
| `large-v3-turbo` (CPU) | 2919 ms | unusable for dictation |

Worth testing beyond Whisper:

| model | why it is interesting |
|---|---|
| **Moonshine** (Useful Sensors) | built for short-form; **no 30-second padding** — cost scales with actual audio length. Directly fixes the problem that makes turbo 2919 ms. |
| **NVIDIA Parakeet / Canary** | strongest open streaming ASR; CTC/TDT, frame-synchronous |
| **wav2vec2 / Conformer-CTC** | one forward pass, no autoregressive loop |
| **distil-whisper** | drop-in, also usable as a speculative draft model |

**Key insight:** CTC models lose punctuation and casing — **which stops mattering once an LLM formats downstream.** A worse-but-faster ASR becomes viable precisely because component 5 exists.

**Measure:** latency, WER on your own voice, behaviour on your jargon.
**Expected impact: HIGH** — but interacts with component 3. Measure them together.

---

## Component 5 — Runtime and precision

| Runtime | Hardware | Notes |
|---|---|---|
| **faster-whisper / CTranslate2** | CPU, CUDA | no Metal — CPU only on Mac |
| **mlx-whisper** | Apple GPU | 6.5x faster than CPU here (452 vs 2919 ms) |
| **whisper.cpp** | Metal, CPU, CUDA | most portable, aggressive quantization |
| **WhisperKit** | CoreML / ANE | Apple-optimized, may use the Neural Engine |
| **ONNX Runtime** | everything | portability over peak speed |

| Precision | Trade-off |
|---|---|
| float32 | most precise; **hallucinated 28 extra seconds on a real file** |
| int8 | 4x smaller; correct on that same file |
| int4 | smallest; test quality carefully |

**Already learned the hard way:** precision changes the *decoding path*, not just speed. Always check output correctness, never just latency.

**Measure:** latency, memory, and output diff between runtimes on the same audio.
**Expected impact: HIGH on Mac** — the GPU path alone was 6.5x.

---

## Component 6 — Formatting pass

Turning `"um so yeah I think we should uh ship this friday right"` into `"I think we should ship this Friday."`

| Approach | Latency | Quality |
|---|---|---|
| **None** | 0 ms | raw transcript, fillers included |
| **Regex / rules** | <1 ms | strips "um/uh", fixes spacing. Brittle but free. |
| **Small local LLM** (0.3–3B) | ~100–250 ms | the real answer. Qwen3-0.6B, Gemma-3-270M, Llama-3.2-1B |
| **Mid local LLM** (7–8B) | 400 ms+ | better, probably over budget |
| **Large local** (31B) | **>9 min to load** — measured, unusable | |
| **Cloud LLM** | 300–800 ms + network | best quality, worst latency, privacy cost |
| **Fine-tuned small model** | ~100 ms | best long-term: train on your own corrections |

**The failure mode is over-editing** — the model "improving" your meaning. Measure this explicitly; it is worse than leaving fillers in.

**Axes to try:** model size, prompt wording, few-shot examples, constrained decoding, temperature 0, streaming the output token-by-token so text appears progressively.

**Measure:** latency, over-edit rate, meaning-error rate.
**Expected impact: HIGHEST on perceived accuracy.** This is what makes it feel magic.

---

## Component 7 — Personalization

Highest perceived-accuracy gain per unit of effort. Generic models mangle names and jargon, and users notice that far more than a 1% WER difference.

| Approach | How | Cost |
|---|---|---|
| **initial_prompt biasing** | pass a vocabulary list as decoder context | free, weak-to-moderate effect |
| **Hotword boosting** | boost logits for specific tokens during decode | medium; needs decoder access |
| **Post-hoc fuzzy match** | correct output against a known-names list | easy, surprisingly effective |
| **LLM-side correction** | give the formatting LLM your name list | easy, combines with component 6 |
| **LoRA fine-tune** | train on your own audio + corrections | best, needs data collection |
| **Learn from edits** | when the user fixes text, remember it | the compounding one |

**Measure:** accuracy on a held-out list of your own names, jargon, and project terms.
**Expected impact: HIGH on perceived quality, LOW on WER.** The two diverge here — which is the point.

---

## Component 8 — Injection

| Approach | Notes |
|---|---|
| Clipboard + synthetic Cmd-V | easiest, clobbers the clipboard |
| CGEvent keystrokes (macOS) | types character by character, can be slow for long text |
| Accessibility API | most correct, most permission friction |

**Measure:** injection latency, per-app failures (terminals, Electron apps, and browsers all behave differently).
**Expected impact: LOW on latency, HIGH on "does it work everywhere".**

---

## Component 9 — Pipelining (free latency)

Not a component — an arrangement. Overlap the stages instead of running them in sequence.

```
sequential:  [--- capture ---][--- ASR ---][-- LLM --][inject]
overlapped:  [--- capture ---]
                  [--- ASR (running live) ---]
                                   [-- LLM on partial --]
                                              [inject]
```

**Ideas to test:** start ASR while recording; start the LLM on the first stable segment; stream LLM output into the app so text appears progressively; pre-warm all models at app launch so cold start never hits the user.

**Expected impact: HIGH, and cheap** — it is scheduling, not new models.

---

## Experiment protocol

1. **Fix everything except one component.**
2. **Warm up**, then run 5+ times, report median AND spread.
3. **Record latency AND quality.** The float32 lesson: a config can be faster and wrong.
4. **If runs disagree, that disagreement is the finding.** Chase it before recording a number.
5. **One row per experiment** in a results table. The table is the deliverable.

### Suggested results table

| # | component changed | setting | t_release→text p50 | p95 | WER | over-edit | notes |
|---|---|---|---|---|---|---|---|
| 0 | baseline | batch, tiny.en, no LLM | | | | | |

---

## Suggested order

1. **v1 skeleton** — push-to-talk, batch, `tiny.en`, no LLM, clipboard inject. Establishes the baseline and makes latency concrete.
2. **Add the LLM pass** with a small model. Biggest perceived-quality jump.
3. **Add pipelining.** Free latency, no new models.
4. **Try streaming** (component 3). The real work, biggest remaining win.
5. **Swap ASR models** — Moonshine and Parakeet especially, since both attack the padding problem directly.
6. **Personalization.** Compounding returns once the rest is stable.

**Rationale for the order:** each step is independently useful, and each makes the next step's measurement more meaningful. Do not start with streaming — build the harness first, or you will not be able to tell whether streaming helped.
