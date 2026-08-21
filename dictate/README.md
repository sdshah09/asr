# dictate

A local dictation tool. Hold a key, talk, text appears in whatever app has focus.
Same shape as Wispr Flow, built from the components mapped out in
[../EXPERIMENTS.md](../EXPERIMENTS.md).

Fully local — no API key, no network.

## Milestones

| # | Milestone | Status | Rust concepts it introduces |
|---|---|---|---|
| 0 | Read a WAV file, print what's inside | done | cargo, crates, `Option`, `Result`, borrowing |
| 1 | Transcribe that file, print text + timing | next | calling C libraries, build scripts |
| 2 | Record from the mic | | threads, `Arc<Mutex>` |
| 3 | Push-to-talk hotkey | | callbacks |
| 4 | Inject text into the focused app | | OS APIs, permissions |
| 5 | VAD | | |
| 6 | LLM cleanup pass | | async |
| 7 | Personalization / custom vocabulary | | |

CLI first. A window is component 8 and comes much later, if ever — the
interesting parts are audio, inference and latency.

## Run

```bash
cd dictate
cargo run -- ../hardset/04_noisy.wav
```

## Why Rust

- **No garbage collector.** The mic delivers a buffer every ~10 ms and the
  callback must return before the next one. A GC pause drops frames.
- **Single binary.** No interpreter to ship, no per-platform wheels.
- **Native OS APIs.** Global hotkeys, accessibility and keystroke injection are
  all C-level calls; Rust's FFI is zero-cost.
- **The engines are already C++.** whisper.cpp and ONNX bind directly.
