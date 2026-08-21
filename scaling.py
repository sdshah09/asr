"""E1.1 — does transcription cost scale with audio length?

Whisper pads every input to 30s. Moonshine does not. Plot duration vs time.
"""
import subprocess, time
from pathlib import Path
from statistics import median
import numpy as np, torch

DURS = [1, 2, 5, 10, 20, 30]
D = Path("scaleset"); D.mkdir(exist_ok=True)
TEXT = ("Speech recognition converts spoken language into written text. "
        "Modern systems use deep neural networks trained on very large datasets. "
        "The encoder reads the audio and the decoder writes the words one at a time. "
        "This process repeats until the model decides the utterance is complete. "
        "Longer recordings contain more words and therefore require more decoding steps. "
        "Engineers measure this cost using the real time factor. ")

# ---- build clips -------------------------------------------------------
src = D / "_src.aiff"
if not src.exists():
    subprocess.run(["say", "-o", str(src), TEXT * 3], check=True)
clips = []
for d in DURS:
    f = D / f"{d:02d}s.wav"
    if not f.exists():
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                        "-t", str(d), "-ar", "16000", "-ac", "1", str(f)], check=True)
    clips.append((d, f))

def bench(fn, path, reps=3):
    fn(path)                                    # warm
    return median([_t(fn, path) for _ in range(reps)])

def _t(fn, path):
    s = time.perf_counter(); fn(path); return time.perf_counter() - s

# ---- whisper (faster-whisper / CTranslate2, CPU) -----------------------
from faster_whisper import WhisperModel
wm = WhisperModel("tiny.en", device="cpu", compute_type="int8", cpu_threads=10)
def run_whisper(p):
    segs, _ = wm.transcribe(str(p), beam_size=1, language="en")
    return "".join(s.text for s in segs)

# ---- moonshine (transformers / torch, CPU) -----------------------------
from transformers import AutoProcessor, MoonshineForConditionalGeneration
from mlx_whisper import audio as A
proc = AutoProcessor.from_pretrained("moonshine-ai/moonshine-tiny")
mm = MoonshineForConditionalGeneration.from_pretrained("moonshine-ai/moonshine-tiny").eval()
def run_moonshine(p):
    wav = np.array(A.load_audio(str(p)))
    inp = proc(wav, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        ids = mm.generate(**inp, max_new_tokens=256)
    return proc.batch_decode(ids, skip_special_tokens=True)[0]

# ---- run ---------------------------------------------------------------
rows = []
for d, f in clips:
    w = bench(run_whisper, f)
    m = bench(run_moonshine, f)
    wc = len(run_whisper(f).split())
    rows.append((d, w, m, wc))
    print(f"  {d:>2}s done   whisper {w*1000:>6.0f}ms   moonshine {m*1000:>6.0f}ms   ({wc} words)")

print(f"\n{'audio':>6}{'words':>7}{'whisper':>11}{'moonshine':>12}{'w RTF':>9}{'m RTF':>9}")
print("-" * 56)
for d, w, m, wc in rows:
    print(f"{d:>5}s{wc:>7}{w*1000:>10.0f}ms{m*1000:>11.0f}ms{w/d:>9.3f}{m/d:>9.3f}")

# ---- plot --------------------------------------------------------------
hi = max(max(w for _, w, _, _ in rows), max(m for _, _, m, _ in rows))
W = 52
print(f"\ntime vs audio length   (W=whisper tiny.en, M=moonshine-tiny)   full scale = {hi*1000:.0f}ms\n")
for d, w, m, _ in rows:
    lw, lm = int(w / hi * W), int(m / hi * W)
    print(f"{d:>3}s W |{'#' * lw}{' ' * (W - lw)}| {w*1000:>6.0f}ms")
    print(f"     M |{'=' * lm}{' ' * (W - lm)}| {m*1000:>6.0f}ms")

f1, f30 = rows[0], rows[-1]
print(f"\n1s -> 30s scaling:")
print(f"  whisper   {f1[1]*1000:.0f}ms -> {f30[1]*1000:.0f}ms   = {f30[1]/f1[1]:.2f}x  (audio grew 30x)")
print(f"  moonshine {f1[2]*1000:.0f}ms -> {f30[2]*1000:.0f}ms   = {f30[2]/f1[2]:.2f}x")
